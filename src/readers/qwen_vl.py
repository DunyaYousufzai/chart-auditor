"""Qwen3-VL chart reader.

Two back ends with the same interface:
  QwenVLLMReader  vLLM, for UBELIX GPUs. Fast batched sampling: n readings per chart in one call.
  QwenHFReader    plain transformers. Slower; useful to test on a laptop with the 2B model.

task="read"  -> the model writes the table (our pipeline)
task="judge" -> the model sees chart + paper table and answers MATCH / MISMATCH (baseline B0)
"""
import base64
import io

from PIL import Image

from readers.prompts import JUDGE_PROMPT, READ_PROMPT

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def _prompt(task, table_text=None):
    return READ_PROMPT if task == "read" else JUDGE_PROMPT.format(table=table_text)


def _data_url(path, max_side=None):
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:  # "smaller images" experiment
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class QwenVLLMReader:
    def __init__(self, model=DEFAULT_MODEL, max_new_tokens=1024, max_model_len=8192,
                 gpu_memory_utilization=0.85, quantization=None, max_side=None, **_):
        from vllm import LLM
        self.llm = LLM(model=model, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       limit_mm_per_prompt={"image": 1}, quantization=quantization,
                       seed=0)
        self.max_new_tokens = max_new_tokens
        self.max_side = max_side
        self.name = model

    def read(self, image_paths, n=1, temperature=0.0, seed=0, task="read", tables=None):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature if n > 1 or temperature > 0 else 0.0,
                            top_p=0.95 if temperature > 0 else 1.0,
                            max_tokens=self.max_new_tokens, seed=seed)
        convs = []
        for i, p in enumerate(image_paths):
            text = _prompt(task, tables[i] if tables else None)
            convs.append([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_url(p, self.max_side)}},
                {"type": "text", "text": text}]}])
        outs = self.llm.chat(convs, sp, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outs]


class QwenHFReader:
    def __init__(self, model=DEFAULT_MODEL, max_new_tokens=1024, device_map="auto", max_side=None, **_):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype, device_map=device_map)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.max_side = max_side
        self.name = model

    def read(self, image_paths, n=1, temperature=0.0, seed=0, task="read", tables=None):
        self.torch.manual_seed(seed)
        results = []
        for i, p in enumerate(image_paths):
            img = Image.open(p).convert("RGB")
            if self.max_side and max(img.size) > self.max_side:
                img.thumbnail((self.max_side, self.max_side))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": _prompt(task, tables[i] if tables else None)}]}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
            sample = n > 1 or temperature > 0
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                          do_sample=sample, temperature=temperature if sample else None,
                                          top_p=0.95 if sample else None, num_return_sequences=n)
            new = out[:, inputs["input_ids"].shape[1]:]
            results.append(self.processor.batch_decode(new, skip_special_tokens=True))
        return results
