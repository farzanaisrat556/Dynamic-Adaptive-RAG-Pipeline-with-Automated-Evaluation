# Adaptive RAG — Query-Complexity Routing

A from-scratch implementation of **Adaptive RAG** (Jeong et al., 2024): instead
of always doing the same retrieval steps for every question, each query is
routed to one of three strategies based on estimated complexity —
**no retrieval**, **single-hop retrieval**, or **multi-hop iterative
retrieval** — and compared against a fair single-retrieval baseline.

> Note: this notebook is designed to run in **Google Colab with a GPU**
> (it loads a 4-bit quantized 7B model). It has not been executed end-to-end
> in this repo's CI/history — read through it before running, especially the
> config cell, if you're adapting it to your own setup.

## What it does

1. **Builds a mixed-complexity evaluation set** (~360 questions) from three
   public datasets, so there's actually something for an adaptive router to
   demonstrate:
   | Tier | Source | Why |
   |---|---|---|
   | A — no retrieval | TriviaQA (`rc.nocontext`) | Well-known facts an LLM usually already knows |
   | B — single-hop | SQuAD v1.1 | Answer is stated directly in one passage |
   | C — multi-hop | HotpotQA (distractor) | Answer requires combining facts across 2+ passages |
2. **Builds one shared FAISS corpus** from the SQuAD/HotpotQA source passages
   — both the baseline and the adaptive pipeline retrieve from the same
   knowledge base (no circular retrieval from the model's own answers).
3. **Loads one LLM** (Mistral-7B-Instruct-v0.2, 4-bit quantized) used by
   every strategy and the router itself — no redefinition, one config.
4. **Routes each question** using a constrained few-shot prompt that outputs
   a single complexity label (A/B/C), then dispatches to the matching
   answer strategy.
5. **Runs a fair baseline** — same LLM, same prompt skeleton, always exactly
   one retrieval step — so any measured difference is attributable to the
   *routing decision itself*, not to a different data source or model.
6. **Checkpoints both runs** to CSV every 20 questions, resuming from where
   it left off if interrupted.
7. **Evaluates accuracy and efficiency** — ROUGE-L, BERTScore, and average
   retrieval calls per question, both overall and broken down by true
   complexity tier — plus router agreement with the gold tier label.
8. **Plots** accuracy and efficiency comparisons.

## What's different from a naive RAG comparison

- A dataset made of only multi-hop questions can never show an adaptive
  router's advantage, because every question would route the same way — the
  three-tier mix here is deliberate.
- The baseline isn't a strawman: it shares the LLM, prompt, and corpus with
  the adaptive pipeline, differing only in always doing one retrieval step.
- The corpus is built once from real source passages, never from the
  model's own generated answers (which would let it reinforce its own
  mistakes over time).

## Requirements

- A Colab GPU runtime (T4 or better) — the LLM alone needs ~5–6GB VRAM in
  4-bit.
- A Google Drive with enough free space for the FAISS index and checkpoint
  CSVs (paths are all under `DRIVE_PATH` in the config cell).

```bash
pip install -r requirements.txt
```

(If running in Colab itself, the notebook's own first cell installs
everything except `torch`/`pandas`/`numpy`/`matplotlib`, which Colab
preinstalls.)

## Usage

1. Open `adaptive_rag.ipynb` in Colab (GPU runtime: **Runtime → Change
   runtime type → GPU**).
2. Run cells top to bottom. Cell 4 is the single source of truth for
   config — change `DRIVE_PATH`, `CHECKPOINT`, `N_PER_TIER`, `TOP_K`, or
   `MAX_HOPS` there, not anywhere else in the notebook.
3. Dataset downloads (TriviaQA/SQuAD/HotpotQA) happen automatically via
   HuggingFace `datasets` — no manual data prep needed.
4. Results land in `results.csv`, `baseline_results.csv`, and
   `adaptive_results.csv` under `DRIVE_PATH`; both are resumable if a run
   gets interrupted (e.g. Colab disconnects).

## Interpreting results

- Look at the **per-tier table**, not just the overall average. A
  well-behaved router should roughly match baseline accuracy on tiers A/B
  while using **fewer retrieval calls** on tier A, and should **beat** the
  baseline on tier C where multi-hop retrieval actually helps.
- If router agreement with the gold tier is low, the LLM-prompt classifier
  is the bottleneck — consider swapping in a small trained classifier (e.g.
  a fine-tuned DistilBERT) as the original Adaptive-RAG paper does.
- Because both pipelines share one corpus, one LLM, and one prompt
  skeleton, any remaining accuracy gap is attributable to the routing
  decision itself.

## Known limitations

- The query-complexity router is a prompted LLM call, not a trained
  classifier — a practical, dependency-light stand-in, but a fine-tuned
  classifier would likely route more reliably (see the original paper's
  labeling procedure).
- `bitsandbytes` 4-bit quantization is Linux/NVIDIA-GPU specific — this
  won't run as-is on CPU-only machines or Apple Silicon.
- Not yet run end-to-end here — treat the config cell (`N_PER_TIER`,
  `MAX_HOPS`, etc.) as a starting point and verify on a small sample before
  a full run, since a full run downloads a 7B model and processes ~360
  questions through it multiple times.

## Reference

Jeong, S. et al. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented
Large Language Models through Question Complexity.*

## License

Add a license of your choice (MIT is a common default) — GitHub can
generate one for you when you create the repo.
