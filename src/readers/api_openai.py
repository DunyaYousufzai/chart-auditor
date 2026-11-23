"""GPT through the OpenAI Responses API (pip install openai). Key: OPENAI_API_KEY.

Same prompt as our Qwen reader. detail: "low", "high", "original" or "auto" (how much of the image
the model sees); "high" by default, "original" is meant for dense images if the model supports it.
"""
import base64
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from readers.prompts import JUDGE_PROMPT, READ_PROMPT


class OpenAIReader:
    def __init__(self, model=None, detail="high", workers=8, **_):
        if not model:
            raise ValueError("give --model (e.g. a current GPT vision model id from the OpenAI docs)")
        self.client = OpenAI()
        self.model, self.detail, self.workers = model, detail, workers
        self.name = model
        self.usage = {}

    def _one(self, path, text):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        try:
            resp = self.client.responses.create(model=self.model, input=[{"role": "user", "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": f"data:image/png;base64,{data}", "detail": self.detail}]}])
        except Exception as e:  # the SDK already retries rate limits and server errors
            print(f"  {path}: {type(e).__name__}: {e}", flush=True)
            return None, None
        u = getattr(resp, "usage", None)
        usage = {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)}
        return resp.output_text, usage

    def read(self, image_paths, n=1, temperature=0.0, seed=0, task="read", tables=None):
        jobs = [(p, READ_PROMPT if task == "read" else JUDGE_PROMPT.format(table=tables[i]))
                for i, p in enumerate(image_paths) for _ in range(n)]
        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(lambda j: self._one(*j), jobs))
        out = []
        for i, p in enumerate(image_paths):
            chunk = res[i * n:(i + 1) * n]
            if any(t is None for t, _ in chunk):
                out.append(None)
                continue
            self.usage[p] = [u for _, u in chunk]
            out.append([t for t, _ in chunk])
        return out
