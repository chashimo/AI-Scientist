import openai
import argparse
import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex

NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20240620",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="semanticscholar",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def check_latex_dependencies():
    """
    Check if required LaTeX dependencies are installed on the system.
    Returns True if all dependencies are found, False otherwise.
    """
    import shutil
    import sys

    required_dependencies = ['pdflatex', 'chktex']
    missing_deps = []

    for dep in required_dependencies:
        if shutil.which(dep) is None:
            missing_deps.append(dep)
    
    if missing_deps:
        print("Error: Required LaTeX dependencies not found:", file=sys.stderr, flush=True)
        return False
    
    return True
    
def worker(
        queue,
        base_dir,
        results_dir,
        model,
        client,
        client_model,
        writeup,
        improvement,
        gpu_id,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.", flush=True)
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            model,
            client,
            client_model,
            writeup,
            improvement,
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}", flush=True)
    print(f"Worker {gpu_id} finished.", flush=True)


def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        client,
        client_model,
        writeup,
        improvement,
        log_file=False,
):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    # Check if baseline_results is a dictionary before extracting means
    if isinstance(baseline_results, dict):
        baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*", flush=True)
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "deepseek-reasoner":
            main_model = Model("deepseek/deepseek-reasoner")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)

        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print_time()
        print(f"*Starting Experiments*", flush=True)
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}", flush=True)
            print(f"Experiments failed for idea {idea_name}", flush=True)
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}", flush=True)
            return False

        print_time()
        print(f"*Starting Writeup*", flush=True)
        ## PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "deepseek-reasoner":
                main_model = Model("deepseek/deepseek-reasoner")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                #perform_writeup(idea, folder_name, coder, client, client_model, engine=args.engine)
                perform_writeup(folder_name, min_score=3)
            except Exception as e:
                print(f"Failed to perform writeup: {e}", flush=True)
                return False
            print("Done writeup", flush=True)
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print_time()

        # Commented out by Hashimoto. 250518.
        # No need to review and improve because my doc simply enumerates detected propaganda candidates.
#        print(f"*Starting Review*", flush=True)
#        ## REVIEW PAPER
#        if writeup == "latex":
#            try:
#                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
#                review = perform_review(
#                    paper_text,
#                    model="gpt-4o-2024-05-13",
#                    client=openai.OpenAI(),
#                    num_reflections=5,
#                    num_fs_examples=1,
#                    num_reviews_ensemble=5,
#                    temperature=0.1,
#                )
#                # Store the review in separate review.txt file
#                with open(osp.join(folder_name, "review.txt"), "w") as f:
#                    f.write(json.dumps(review, indent=4))
#            except Exception as e:
#                print(f"Failed to perform review: {e}", flush=True)
#                return False
#
#        ## IMPROVE WRITEUP
#        if writeup == "latex" and improvement:
#            print_time()
#            print(f"*Starting Improvement*", flush=True)
#            try:
#                perform_improvement(review, coder)
#                generate_latex(
#                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
#                )
#                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
#                review = perform_review(
#                    paper_text,
#                    model="gpt-4o-2024-05-13",
#                    client=openai.OpenAI(),
#                    num_reflections=5,
#                    num_fs_examples=1,
#                    num_reviews_ensemble=5,
#                    temperature=0.1,
#                )
#                # Store the review in separate review.txt file
#                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
#                    f.write(json.dumps(review))
#            except Exception as e:
#                print(f"Failed to perform improvement: {e}", flush=True)
#                return False
        return True
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}", flush=True)
        return False
    finally:
        print("FINISHED IDEA", flush=True)
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


if __name__ == "__main__":
    print("[DEBUG] Starting main()", file=sys.stderr, flush=True)
    args = parse_arguments()
    print(f"[DEBUG] Parsed args: {args}", file=sys.stderr, flush=True)

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}.", flush=True)
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}", flush=True)

    # Check LaTeX dependencies before proceeding
    if args.writeup == "latex" and not check_latex_dependencies():
        sys.exit(1)

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )
    if not args.skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
            engine=args.engine,
        )

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)
    
    #novel_ideas = [idea for idea in ideas if idea["novel"]]
    # novel_ideas = list(reversed(novel_ideas))
    # Modified by Hashimoto
    if args.skip_novelty_check:
        novel_ideas = ideas
    else:
        novel_ideas = [idea for idea in ideas if idea.get("novel", False)]

    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes", flush=True)
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                    gpu_id,
                ),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel processes completed.", flush=True)
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}", flush=True)
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}", flush=True)
            except Exception as e:
                print(f"Failed to evaluate idea {idea['Name']}: {str(e)}", flush=True)
                import traceback
                print(traceback.format_exc())
    print("All ideas evaluated.", flush=True)
