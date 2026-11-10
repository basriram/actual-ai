"""
Decision-model server for actual-ai (fork of sakowicz/actual-ai).

Serves an OpenAI-compatible /v1/chat/completions endpoint backed by an
Unsloth FastDecisionModel (JEV-style) trained on the household's Quicken
transactions. actual-ai points at this via:

  LLM_PROVIDER=openai
  OPENAI_BASE_URL=http://<spark-host>:8100/v1
  OPENAI_API_KEY=local-decision
  OPENAI_MODEL=quicken-decision

Contract with actual-ai:
- actual-ai renders PROMPT_TEMPLATE (see prompt-template.hbs) so the user
  message IS the decision-model "state" in the exact training format, and
  the category list lines carry  `* Name (ID: "id")`  pairs.
- This server builds the choice criteria from those lines, scores every
  category with the decision head (no text generation), maps the winning
  category name back to its Actual category ID, and replies with the JSON
  object actual-ai expects: {"type": "existing", "categoryId": "..."}.
"""
import json
import os
import re
import threading

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODEL_DIR = os.environ.get("DECISION_MODEL_DIR", "/models/quicken-decision-lora")
MAX_SEQ_LENGTH = int(os.environ.get("DECISION_MAX_SEQ_LENGTH", "6144"))
HOST = os.environ.get("DECISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("DECISION_PORT", "8100"))

# `* Category Name (ID: "cat-uuid")` lines rendered by prompt-template.hbs
CATEGORY_LINE = re.compile(r'^\*\s*(.+?)\s*\(ID:\s*"([^"]+)"\)\s*$', re.M)

app = FastAPI()
_state = {"model": None, "tokenizer": None}


class ChatRequest(BaseModel):
    messages: list
    model: str = "quicken-decision"
    # sampling params are accepted and ignored: the decision head scores
    # choices directly, no text generation happens.


def load_model():
    if _state["model"] is not None:
        return _state["model"], _state["tokenizer"]
    from unsloth import FastDecisionModel
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=MODEL_DIR, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)
    FastDecisionModel.for_inference(model)
    _state["model"], _state["tokenizer"] = model, tokenizer
    return model, tokenizer


def parse_state_and_categories(prompt: str):
    """Split the rendered prompt into decision state + name->id map."""
    pairs = {name.strip(): cid for name, cid in CATEGORY_LINE.findall(prompt)}
    if not pairs:
        raise ValueError("prompt carries no `* Name (ID: \"...\")` category lines")
    # The state is everything before the "Existing categories by group:" marker
    marker = "Existing categories by group:"
    state = prompt.split(marker)[0].strip()
    return state, pairs


def run_blocking(fn):
    """FastAPI sync def endpoints already run in a threadpool; model calls are
    single-threaded per process, so serialize them."""
    with _LOCK:
        return fn()


_LOCK = threading.Lock()


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = next((m["content"] for m in reversed(req.messages)
                   if m.get("role") == "user"), "")
    model, tokenizer = load_model()
    try:
        state, name_to_id = run_blocking(lambda: parse_state_and_categories(prompt))
        questions = {
            "category": {
                "type": "choice",
                "instructions": (
                    "You are a household budget transaction classifier. Given a "
                    "bank/credit card transaction, output exactly one category "
                    "label from the household's Quicken taxonomy."
                ),
                "criteria": {name: name for name in name_to_id},
            }
        }
        answers = run_blocking(lambda: FastDecisionModel_predict(
            model, tokenizer, state, questions))
        answer, probs = answers["category"]["answer"], answers["category"]["probabilities"]
        # Transfers: Actual Budget handles account-to-account transfers
        # natively. TRANSFER_ACTION=skip leaves them unclassified (marked
        # "not guessed" upstream, never re-processed) to avoid
        # double-counting; the default "categorize" assigns the Transfer
        # category ID so they stay visible in Actual.
        if answer.startswith("Transfer") and \
                os.environ.get("TRANSFER_ACTION", "categorize") == "skip":
            content = {"type": "existing", "categoryId": "",
                       "_skipped": answer}
        else:
            content = {"type": "existing", "categoryId": name_to_id[answer],
                       "_confidence": probs.get(answer)}
    except Exception as e:  # noqa: BLE001 - respond as "not guessed"
        content = {"type": "existing", "categoryId": "", "_error": str(e)}
    body = {
        "id": "chatcmpl-decision",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": json.dumps(content)}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return body


def FastDecisionModel_predict(model, tokenizer, state, questions):
    from unsloth import FastDecisionModel
    return FastDecisionModel.predict(model, tokenizer, state, questions)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "quicken-decision",
                                        "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR, "loaded": _state["model"] is not None}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
