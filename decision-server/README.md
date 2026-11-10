# Decision-model backend (JEV-style) for actual-ai

This fork replaces the hosted-LLM categorization with a locally trained
Unsloth **decision model** (Qwen3.5-4B + decision head, LoRA) trained on the
household's categorized Quicken transactions. Training follows
https://machinelearningmastery.com/how-to-train-your-own-decision-model-like-jev-using-unsloth/

## Architecture

```
Actual Budget --> actual-ai (this fork)
                    |  LLM_PROVIDER=openai, OPENAI_BASE_URL=http://spark:8100/v1
                    v
             decision-server/server.py  (OpenAI-compatible shim, this dir)
                    |  FastDecisionModel.predict() over the category list
                    v
             Qwen3.5-4B + LoRA + decision head  (on the DGX Spark)
```

No text generation happens at inference: the decision head scores every
category in the Actual category list and the winner's name is mapped back to
its Actual category ID.

## Fork changes vs upstream

1. `decision-server/` - OpenAI-compatible shim (`server.py`) + `prompt-template.hbs`
   that renders transactions in the exact training state format
   (`Payee/Memo/Amount/Account type/Date`) and category lines
   `* Name (ID: "...")` the shim parses into choice criteria.
2. `src/config.ts` - `PROMPT_TEMPLATE_FILE` env var: read the template from a
   file path instead of inlining it in the environment.
3. Docs (this file).

## Training (one-time, on the DGX Spark)

See `~/quicken-decision/` on 192.168.60.139 (`train_decision.py` +
`convert_to_decision.py`, dataset distilled from Quicken by the finance
agent). Artifacts saved with `model.save_pretrained("quicken-decision-lora")`.

## Serving

On the Spark:

```bash
pip install fastapi uvicorn   # into the unsloth venv
DECISION_MODEL_DIR=~/quicken-decision/quicken-decision-lora \
  ~/quicken-decision/.venv/bin/python decision-server/server.py
```

## actual-ai side (docker-compose)

```yaml
  actual-ai:
    image: local/actual-ai-decision:latest
    environment:
      ACTUAL_SERVER_URL: http://actual_server:5006
      ACTUAL_PASSWORD: ...
      ACTUAL_BUDGET_ID: ...
      LLM_PROVIDER: openai
      OPENAI_BASE_URL: http://192.168.60.139:8100/v1
      OPENAI_API_KEY: local-decision
      OPENAI_MODEL: quicken-decision
      PROMPT_TEMPLATE_FILE: /app/decision-server/prompt-template.hbs
      FEATURES: '["classifyOnStartup", "syncAccountsBeforeClassify"]'  # dryRun on by default
    volumes:
      - ./decision-server/prompt-template.hbs:/app/decision-server/prompt-template.hbs:ro
```

Notes / caveats:
- Training states always said `Account type: creditcard`; the template pins
  that value. If checking-account classification underperforms, retrain with
  per-account type data.
- Rules are intentionally omitted from the template (model was trained
  without them); Actual's rule-match strategy still runs upstream of the LLM.
- `suggestNewCategories` needs text generation the decision head can't do;
  leave it off, or fall back to a hosted LLM for that feature.
