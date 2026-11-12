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
- The decision head is ALWAYS scored against the 199 labels it was trained
  on (dataset/labels.json). Scoring against Actual's own category names
  drops top-1 from ~82% to ~59%: 387 flat names full of near-synonyms
  (Dining vs Restaurants, Print vs Printing (Business)) poison the choice
  set. The winning training label is then bridged to the Actual category
  ID by exact match, falling back to leaf-name match.
"""
import json
import os
import re
import threading

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODEL_DIR = os.environ.get("DECISION_MODEL_DIR", "/models/quicken-decision-lora")
LABELS_FILE = os.environ.get("DECISION_LABELS",
                             "/home/sriram/quicken-decision/dataset/labels.json")
CATEGORY_MAP_FILE = os.environ.get(
    "DECISION_CATEGORY_MAP",
    "/home/sriram/quicken-decision/dataset/category_map.json")
MAX_SEQ_LENGTH = int(os.environ.get("DECISION_MAX_SEQ_LENGTH", "6144"))
HOST = os.environ.get("DECISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("DECISION_PORT", "8100"))

# `* Category Name (ID: "cat-uuid")` lines rendered by prompt-template.hbs
CATEGORY_LINE = re.compile(r'^\*\s*(.+?)\s*\(ID:\s*"([^"]+)"\)\s*$', re.M)

# Payee/memo patterns that identify transfers: they are not a classification
# problem and Actual handles them natively. ~9% of this household's volume.
TRANSFER_PATTERN = re.compile(
    r"PAYMENT THANK|TRANSFER|BILL PAY|AUTOPAY|AUTOBANK", re.I)

# Income/fee lines that merely contain "Transfer" in their payee - they must
# reach the model (interest income, deductible bank fees, securities moves).
TRANSFER_EXEMPT = re.compile(r"INTEREST|FEE|SEC TRANSFER", re.I)

app = FastAPI()
_state = {"model": None, "tokenizer": None, "labels": None, "map": None}


def training_labels():
    """The 199 labels the decision head was trained on.

    The choice set must stay exactly as trained: scoring against Actual's
    flat category names (387 of them, full of near-synonyms like Dining vs
    Restaurants) drops top-1 from 82.5% to 59% - classic distractor
    poisoning. Predictions are then bridged to Actual IDs by leaf name.
    """
    if _state["labels"] is None:
        _state["labels"] = json.load(open(LABELS_FILE))
    return _state["labels"]


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
    """Split the rendered prompt into decision state + [(name, id)] pairs.

    A list, not a dict: Actual budgets can contain two categories with the
    same name (different groups), and a dict would silently collapse them.
    """
    pairs = [(name.strip(), cid) for name, cid in CATEGORY_LINE.findall(prompt)]
    if not pairs:
        raise ValueError("prompt carries no `* Name (ID: \"...\")` category lines")
    # The state is everything before the "Existing categories by group:" marker
    marker = "Existing categories by group:"
    state = prompt.split(marker)[0].strip()
    return state, pairs


def category_map():
    """Quicken training label -> Actual category name (explicit alias table).

    Actual's taxonomy is nothing like the training one: "Dining Out" vs
    "Food & Dining : Restaurants", "Transfers", "Credit Card Payments".
    Leaf-name matching alone misses most of it, so an explicit map (with
    null = leave uncategorized) is the authoritative bridge.
    """
    if _state["map"] is None:
        try:
            _state["map"] = json.load(open(CATEGORY_MAP_FILE))
        except FileNotFoundError:
            _state["map"] = {}
    return _state["map"]


def resolve_category(name, pairs):
    """Map a predicted label to an Actual category ID.

    1. Explicit alias map (category_map.json): label -> Actual category name.
       A null mapping means "leave uncategorized" (e.g. Transfers).
    2. Exact name match, then leaf-name match, as fallback for labels the
       map does not cover.

    Returns (category_id, ambiguous).
    """
    cmap = category_map()
    if name in cmap:
        target = cmap[name]
        if target is None:
            return None, False
        for full, cid in pairs:
            if full == target:
                return cid, False
        leaf = target.split(" : ")[-1].strip().lower()
        matches = [cid for full, cid in pairs
                   if full.split(" : ")[-1].strip().lower() == leaf]
        if matches:
            return matches[0], len(matches) > 1
        return None, False

    exact = [cid for full, cid in pairs if full == name]
    if exact:
        return exact[0], len(exact) > 1
    leaf = name.split(" : ")[-1].strip()
    leaf_matches = [cid for full, cid in pairs
                    if full.split(" : ")[-1].strip() == leaf]
    if not leaf_matches:
        return None, False
    return leaf_matches[0], len(leaf_matches) > 1


# Categories whose misclassification has tax/audit consequences (business
# expenses feed Schedule K-1 reimbursements, charity/medical feed itemized
# deductions). Never auto-post these regardless of model confidence.
TAX_SENSITIVE_PREFIXES = ("Business", "Charity", "Medical", "Insurance", "Tax")


def is_tax_sensitive(name):
    leaf = name.split(" : ")[-1].strip()
    return any(leaf.startswith(p) for p in TAX_SENSITIVE_PREFIXES)


def run_blocking(fn):
    """FastAPI sync def endpoints already run in a threadpool; model calls are
    single-threaded per process, so serialize them."""
    with _LOCK:
        return fn()


_LOCK = threading.Lock()


def transfer_like(state):
    """Deterministic transfer detection on payee/memo, before any GPU work.

    ~9% of this household's transactions are transfers; they are not a
    classification problem, and Actual handles them natively. Matching here
    saves the model call entirely and keeps logs clean.

    Exemptions matter: "Transfer Interest From CD Account" is interest
    income and "Online External Transfer Fee" is a deductible bank fee -
    those must reach the model. Counterparty-account payees (e.g. "Common
    Chase Checking") do NOT match here; they are caught post-prediction by
    the Transfer rule response.
    """
    m = re.search(r"^Payee:\s*(.+)$", state, re.M)
    memo = re.search(r"^Memo:\s*(.+)$", state, re.M)
    text = " ".join(x.group(1) for x in (m, memo) if x)
    if TRANSFER_EXEMPT.search(text):
        return False
    return bool(TRANSFER_PATTERN.search(text))


def chat_response(content, req):
    return {
        "id": "chatcmpl-decision",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": json.dumps(content)}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = next((m["content"] for m in reversed(req.messages)
                   if m.get("role") == "user"), "")
    # Transfer prefilter: no model load, no GPU time, clean rule response.
    if os.environ.get("TRANSFER_PREFILTER", "on") == "on":
        state_pre = prompt.split("Existing categories by group:")[0].strip()
        if transfer_like(state_pre):
            return chat_response({"type": "rule",
                                  "ruleName": "decision: transfer (leave uncategorized)"},
                                 req)
    model, tokenizer = load_model()
    try:
        state, pairs = run_blocking(lambda: parse_state_and_categories(prompt))
        questions = {
            "category": {
                "type": "choice",
                "instructions": (
                    "You are a household budget transaction classifier. Given a "
                    "bank/credit card transaction, output exactly one category "
                    "label from the household's Quicken taxonomy."
                ),
                "criteria": {label: label for label in training_labels()},
            }
        }
        answers = run_blocking(lambda: FastDecisionModel_predict(
            model, tokenizer, state, questions))
        answer, probs = answers["category"]["answer"], answers["category"]["probabilities"]
        # Transfers predicted by the head (counterparty-account payees that
        # the prefilter cannot catch) and labels with no matching Actual
        # category are not writable by definition - answer with a rule
        # response instead of an empty categoryId that makes actual-ai
        # throw. TRANSFER_ACTION=categorize tries to map a real Transfer
        # category first, for budgets that have one.
        if answer.startswith("Transfer"):
            cid, _ = resolve_category(answer, pairs) if \
                os.environ.get("TRANSFER_ACTION", "skip") == "categorize" else (None, False)
            if cid:
                content = {"type": "existing", "categoryId": cid,
                           "_confidence": probs.get(answer)}
            else:
                content = {"type": "rule",
                           "ruleName": "decision: transfer (leave uncategorized)"}
        elif os.environ.get("TAX_GUARD", "on") == "on" and is_tax_sensitive(answer):
            # Tax-sensitive classes are routed to the review queue even when
            # confidence is high: 76% exact accuracy on that slice is not
            # good enough for deduction-relevant lines.
            content = {"type": "existing", "categoryId": "",
                       "_review": "tax_sensitive", "_predicted": answer,
                       "_confidence": probs.get(answer)}
        else:
            cid, ambiguous = resolve_category(answer, pairs)
            if cid is None:
                content = {"type": "rule",
                           "ruleName": f"decision: no matching category ({answer})"}
            else:
                content = {"type": "existing", "categoryId": cid,
                           "_confidence": probs.get(answer)}
                if ambiguous:
                    content["_ambiguous"] = True
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
