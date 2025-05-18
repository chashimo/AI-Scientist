import os
import os.path as osp
import glob
import json
import re
import argparse
import shutil
import subprocess

def escape_latex(text):
    if not isinstance(text, str):
        return text
    repls = [
        ("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"),
        ("$", r"\$"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
        ("_", r"\_"), ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
    ]
    for a, b in repls:
        text = text.replace(a, b)
    text = re.sub(r"(?<!-)-(?!-)", "-", text)
    return text

def load_results(result_dirs, min_score=3):
    groups = dict()
    for run_dir in result_dirs:
        jsonl_path = osp.join(run_dir, "results_propaganda.jsonl")
        if not osp.exists(jsonl_path):
            continue
        with open(jsonl_path) as f:
            for line in f:
                obj = json.loads(line)
                if int(obj.get("score", 0)) < min_score:
                    continue
                nar = obj["narrative"]
                vid = obj["video"]
                entry = {
                    "title": vid["title"],
                    "url": vid["url"],
                    "score": obj["score"]
                }
                groups.setdefault(nar, []).append(entry)
    for n in groups:
        groups[n] = sorted(groups[n], key=lambda v: -v["score"])
    return groups

def make_latex_blocks(groups):
    blocks = []
    for idx, (narrative, videos) in enumerate(groups.items()):
        if idx > 0:
            blocks.append(r"\vspace{1em}")
        blocks.append(
            r"""\noindent
\begin{tabular}{p{0.98\textwidth}}
\textbf{Propaganda Narrative:} %s \\
(English: ) \\
\end{tabular}
""" % escape_latex(narrative)
        )
        for v in videos:
            blocks.append(
                r"""\noindent
\begin{tabular}{p{0.98\textwidth}}
Title: %s \\
(English: ) \\
URL: \url{%s} \\
Score: %s \\
\end{tabular}
\vspace{0.5em}
""" % (escape_latex(v["title"]), v["url"], v["score"])
            )
    return "\n".join(blocks)

def generate_latex(latex_dir, pdf_out, timeout=60):
    print("Running xelatex...")
    try:
        subprocess.run(["xelatex", "-interaction=nonstopmode", "template.tex"], cwd=latex_dir, timeout=timeout)
        subprocess.run(["xelatex", "-interaction=nonstopmode", "template.tex"], cwd=latex_dir, timeout=timeout)
        shutil.copy(osp.join(latex_dir, "template.pdf"), pdf_out)
        print("PDF copied to", pdf_out)
    except Exception as e:
        print("LaTeX compile failed:", e)

def perform_writeup(folder, min_score=3):
    run_dirs = sorted(glob.glob(osp.join(folder, "run_*")))
    groups = load_results(run_dirs, min_score=min_score)
    latex_body = make_latex_blocks(groups)
    print("--- Preview of LaTeX body ---")
    print(latex_body[:3000] + ("..." if len(latex_body) > 3000 else ""))
    print("-----------------------------")

    latex_dir = osp.join(folder, "latex")
    tex_path = osp.join(latex_dir, "template.tex")
    with open(tex_path, "r") as f:
        tex = f.read()
    tex = tex.replace("% KEY_FINDINGS_PLACEHOLDER", latex_body)
    with open(tex_path, "w") as f:
        f.write(tex)
    generate_latex(latex_dir, osp.join(folder, "propaganda_report.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to experiment result folder")
    parser.add_argument("--min_score", type=int, default=3, help="Minimum score threshold")
    args = parser.parse_args()
    perform_writeup(args.folder, min_score=args.min_score)

