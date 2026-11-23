"""Gemini through the Google GenAI SDK (pip install google-genai). Key: GEMINI_API_KEY.

Same prompt as our Qwen reader. Uses client.models.generate_content with the image as inline bytes.
If your SDK version only documents client.interactions.create, adapt _one() (see the Gemini docs).
"""
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

from readers.prompts import JUDGE_PROMPT, READ_PROMPT


class GeminiReader:
    def __init__(self, model=None, workers=8, **_):
        if not model:
            raise ValueError("give --model (e.g. a current Gemini model id from the Gemini docs)")
        self.client = genai.Client()
        self.model, self.workers = model, workers
        self.name = model
        self.usage = {}

    def _one(self, path, text):
        with open(path, "rb") as f:
            data = f.read()
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=[types.Part.from_bytes(data=data, mime_type="image/png"), text])
        except Exception as e:
            print(f"  {path}: {type(e).__name__}: {e}", flush=True)
            return None, None
        u = getattr(resp, "usage_metadata", None)
        usage = {"input_tokens": getattr(u, "prompt_token_count", None),
                 "output_tokens": getattr(u, "candidates_token_count", None),
                 "thinking_tokens": getattr(u, "thoughts_token_count", None)}
        return resp.text or "", usage

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
