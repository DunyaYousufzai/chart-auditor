"""DePlot (google/deplot, Pix2Struct, Apache-2.0): baseline chart -> table reader.

Runs on CPU (slow, ~10-30 s per chart) or GPU. Output looks like
"TITLE | Model accuracy <0x0A> Model | Accuracy <0x0A> ResNet | 82.4 ..."
which tables.parse_model_output understands.
"""
import torch
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

PROMPT = "Generate underlying data table of the figure below:"


class DePlotReader:
    def __init__(self, model="google/deplot", device=None, max_new_tokens=512, batch_size=4, **_):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = Pix2StructProcessor.from_pretrained(model)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model, dtype=dtype).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.name = model

    @torch.inference_mode()
    def read(self, image_paths, n=1, temperature=0.0, seed=0):
        torch.manual_seed(seed)
        results = []
        for start in range(0, len(image_paths), self.batch_size):
            batch = image_paths[start:start + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=images, text=[PROMPT] * len(images),
                                    return_tensors="pt").to(self.device)
            if inputs.get("flattened_patches") is not None:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.model.dtype)
            sample = n > 1 or temperature > 0
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=sample, temperature=temperature if sample else None,
                top_p=0.95 if sample else None, num_return_sequences=n)
            texts = self.processor.batch_decode(out, skip_special_tokens=True)
            for i in range(len(batch)):
                results.append(texts[i * n:(i + 1) * n])
        return results
