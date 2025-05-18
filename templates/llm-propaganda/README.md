# YouTube Anti-Japan Propaganda Detection Template

This template applies the AI Scientist framework to the detection of anti-Japan propaganda on YouTube, focusing on information warfare and hostile narratives. It leverages the YouTube Data API for data collection and uses a language model (LLM) for multi-stage detection: narrative generation, query generation, search, and propaganda scoring.

The project is part of an open-ended research direction exploring how LLMs can be used for automated OSINT (Open-Source Intelligence) tasks in the context of information warfare.

---

## Project Overview

This template demonstrates how LLM-based systems can assist in:

- Generating plausible propaganda narratives that might be used in information warfare against Japan
- Automatically generating search queries (Chinese and English) from those narratives
- Collecting candidate YouTube videos via the YouTube Data API
- Scoring videos using an LLM to assess their relevance as anti-Japan propaganda, with multi-level scoring
- Reporting detected propaganda content

---

## File Structure

- `experiment.py`: Main experiment script that generates narratives, produces search queries, collects YouTube data, and classifies videos using an LLM (with safe API handling).
- `prompt.json`: System and task description for LLM-based idea generation and experimentation guidance.
- `seed_ideas.json`: A list of initial experimental ideas for improving the detection pipeline.
- `latex/`: LaTeX template for generating papers in ICLR format.

---

## Installation

```bash
pip install openai google-api-python-client matplotlib tqdm
```

---

## API Setup

To access YouTube data:

1. Obtain a YouTube Data API key from [Google Cloud Console](https://console.cloud.google.com/).
2. Set the environment variable:

```bash
export YOUTUBE_API_KEY="your-youtube-api-key-here"
```

To access OpenAI models:

```bash
export OPENAI_API_KEY="your-openai-key-here"
```

---

## Running the Experiment

Run the experiment to generate narratives, collect YouTube data, and classify results:

```bash
python experiment.py --out_dir run_0
```

Note: `python plot.py` is not required.

---

## Running AI Scientist

To initiate automatic idea generation, experimentation, and write-up:

```bash
export OPENAI_API_KEY="..."
export YOUTUBE_API_KEY="..."

cd AI-Scientist/

python launch_scientist.py \
	--experiment llm-propaganda \
	--num-ideas 1 \
	--skip-novelty-check \
	--model "gpt-4o" \
	--engine openalex
```

