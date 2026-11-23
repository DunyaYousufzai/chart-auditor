"""ChartGemma (ahmed-masry/chartgemma, PaliGemma 3B, MIT): instruction model trained on charts.

The model card documents QA, summaries and fact checking but no table-extraction prompt, so we use an
instruction picked from four tried on 3 test charts (the only one that gave tables).
"""
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

PROMPT = "Extract the data table of the chart in markdown format."


class ChartGemmaReader:
    def __init__(self, model="ahmed-masry/chartgemma", device=None, max_new_tokens=512, prompt=PROMPT, **_):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.name = model

    @torch.inference_mode()
    def read(self, image_paths, n=1, temperature=0.0, seed=0, **_):
        torch.manual_seed(seed)
        results = []
        for p in image_paths:
            inputs = self.processor(text="<image>" + self.prompt, images=Image.open(p).convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            sample = n > 1 or temperature > 0
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      num_beams=1 if sample else 4, do_sample=sample,
                                      temperature=temperature if sample else None, num_return_sequences=n)
            new = out[:, inputs["input_ids"].shape[1]:]
            results.append(self.processor.batch_decode(new, skip_special_tokens=True))
        return results
