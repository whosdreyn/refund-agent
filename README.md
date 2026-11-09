# refund-agent

A LangChain customer-support agent that handles refund requests, evaluated end to end:
LangSmith datasets and evaluators, an LLM judge, a step-level trajectory engine,
sabotage tests, bootstrap statistics and a CI regression gate.

The trajectory checks, statistics and gate helpers come from [eval-kit](https://github.com/whosdreyn/eval-kit),
which must be cloned next to this repository (`../eval-kit`).

## Layout

```
refund_agent/   the agent: prompts (v1–v6, sabotaged s1–s3), tools, fake data, response schema
evaluation/     everything that evaluates the agent
  ls_evals.py           LangSmith target + evaluators (answer and trajectory)
  config_trajectoire.py DAG, argument provenance rules and safety gates for the engine
  comparer.py           paired bootstrap comparison of two experiments
  deepeval_*.py         the same agent evaluated with DeepEval metrics
data/           scenarios.csv (LangSmith dataset source)
eval.toml       evaluation gate: frozen dataset version, rules, margins and baseline (eval-kit)
scripts/        one-off tools: upload the dataset, print a trajectory
tests/          fast, free unit tests of the trajectory configuration (pytest)
docs/           learning path, method sheet and DeepEval guide
```

## Setup

```bash
git clone https://github.com/whosdreyn/eval-kit
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
| Freeze a dataset version | `uv run python -m scripts.freeze_dataset v2` |
| Run an experiment (new baseline) | `uv run evalkit run` |
| Run the CI gate | `uv run evalkit gate` |

`PROMPT_VERSION` selects the prompt (`v6` by default). The `s1`–`s3` versions are deliberately
faulty and only exist to test the evaluators.

## The gate

The CI gate is `evalkit gate` from [eval-kit](https://github.com/whosdreyn/eval-kit), driven by
`eval.toml`. It evaluates the current code on the frozen dataset version (`refund-scenarios`, tag `v1`,
split `holdout`, 10 scenarios × 3 runs) and blocks the merge when:

- a safety rule fails even once: a ticket created without a refund request, or a trajectory with an
  invented argument or a crossed safety gate;
- a primary score (decision, amount, tools) is significantly worse than the `[baseline]` experiment
  (95% bootstrap confidence interval of the difference below the margin);
- scores are missing (it fails closed).

`message_conforme` is reported but never blocks: its LLM judge is not calibrated against humans yet.
Exit codes: 0 passed, 1 blocked (the agent regressed), 2 the evaluation could not be done.

To change the baseline, run `uv run evalkit run`, check the new experiment, then put its name in
`[baseline]` of `eval.toml` through a reviewed pull request.
