"""Claude through the Anthropic API (pip install anthropic). Key: ANTHROPIC_API_KEY.

Same prompt as our Qwen reader. Current Claude models take no temperature, so every reading is the
model's default sampling; n > 1 just asks n times. Server-side fallbacks are on for the models that
support them: if a request is declined, the API retries it on another model and the served model is
saved in "served_by" (count those charts separately if any appear).
"""
import base64
from concurrent.futures import ThreadPoolExecutor

import anthropic

from readers.prompts import JUDGE_PROMPT, READ_PROMPT

FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2", "claude-sonnet-5-5"}


class ClaudeReader:
    def __init__(self, model="claude-opus-5-5", effort="high", max_new_tokens=16000, workers=8, **_):
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens, self.workers = model, effort, max_new_tokens, workers
        self.name = model
        self.usage = {}

    def _one(self, path, text):
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        kw = {}
        if self.model in FALLBACK_MODELS:
            kw = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=self.max_tokens, output_config={"effort": self.effort},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": text}]}], **kw)
        except anthropic.APIStatusError as e:
            print(f"  {path}: API error {e.status_code}: {e.message}", flush=True)
            return None, None
        except anthropic.APIConnectionError:
            print(f"  {path}: connection error", flush=True)
            return None, None
        out = "" if resp.stop_reason == "refusal" else "".join(b.text for b in resp.content if b.type == "text")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                 "served_by": resp.model, "stop_reason": resp.stop_reason}
        return out, usage

    def read(self, image_paths, n=1, temperature=0.0, seed=0, task="read", tables=None):
        jobs = [(p, READ_PROMPT if task == "read" else JUDGE_PROMPT.format(table=tables[i]))
                for i, p in enumerate(image_paths) for _ in range(n)]
        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(lambda j: self._one(*j), jobs))
        out = []
        for i, p in enumerate(image_paths):
            chunk = res[i * n:(i + 1) * n]
            if any(t is None for t, _ in chunk):
                out.append(None)          # failed: not written, so a rerun retries it
                continue
            self.usage[p] = [u for _, u in chunk]
            out.append([t for t, _ in chunk])
        return out
