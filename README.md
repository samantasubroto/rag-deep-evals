# generator-eval

Evaluation harness for the **generator** stage of RAG (Retrieval-Augmented Generation) systems.
Given a question and fixed retrieved contexts, it measures answer quality, for example
faithfulness (groundedness in the context), answer relevance, and completeness.

## Layout

| Path | Purpose |
|------|---------|
| `src/generator_eval/` | Package source |
| `tests/` | Unit tests |
| `data/raw/` | Source documents (git-ignored) |
| `data/datasets/` | Eval sets (question, contexts, reference answer) |
| `configs/` | YAML run configs |
| `results/` | Run outputs (git-ignored) |
| `notebooks/` | Analysis notebooks |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add ANTHROPIC_API_KEY
```

## Status

Scaffolding only; no implementation yet.
