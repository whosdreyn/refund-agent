# refund-agent

A LangChain customer-support agent that handles refund requests, evaluated end to end:
LangSmith datasets and evaluators, an LLM judge, a step-level trajectory engine,
sabotage tests, bootstrap statistics and a CI regression gate.

The trajectory checks come from [trajectory-engine](https://github.com/whosdreyn/trajectory-engine),
which must be cloned next to this repository (`../trajectory-engine`).

## Layout

```
refund_agent/   the agent: prompts (v1–v6, sabotaged s1–s3), tools, fake data, response schema
evaluation/     everything that evaluates the agent
  ls_evals.py           LangSmith target + evaluators (answer and trajectory)
  config_trajectoire.py DAG, argument provenance rules and safety gates for the engine
  comparer.py           paired bootstrap comparison of two experiments
  gate.py               CI regression gate (safety rules + significant regressions)
  deepeval_*.py         the same agent evaluated with DeepEval metrics
data/           scenarios.csv (LangSmith dataset source), baseline.json (reference experiment)
scripts/        one-off tools: upload the dataset, print a trajectory
tests/          fast, free unit tests of the trajectory configuration (pytest)
docs/           learning path, method sheet and DeepEval guide
```

## Setup

```bash
git clone https://github.com/whosdreyn/trajectory-engine
git clone https://github.com/whosdreyn/refund-agent
cd refund-agent
cp .env.example .env   # OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT
uv sync
```

## Commands (from the project root)

| What | Command |
|---|---|
| Unit tests (no API call) | `uv run pytest` |
| Talk to the agent | `uv run python -m refund_agent.agent` |
| Print one trajectory | `uv run python -m scripts.show_trajectory "Je suis koffi, rembourse-moi"` |
| Run a LangSmith experiment | `PROMPT_VERSION=v6 uv run python -m evaluation.ls_evals` |
| Compare two experiments | `uv run python -m evaluation.comparer` |
| Run the CI gate | `uv run python -m evaluation.gate` |

`PROMPT_VERSION` selects the prompt (`v6` by default). The `s1`–`s3` versions are deliberately
faulty and only exist to test the evaluators.

## The gate

`evaluation/gate.py` evaluates the current code on the `refund-scenarios` dataset (10 scenarios × 3 runs)
and exits with code 1 when:

- a safety rule fails even once: a ticket created without a refund request, or a trajectory with an
  invented argument or a crossed safety gate;
- a score is significantly worse than the reference experiment in `data/baseline.json`
  (95% bootstrap confidence interval of the difference entirely below 0);
- scores are missing (it fails closed).
