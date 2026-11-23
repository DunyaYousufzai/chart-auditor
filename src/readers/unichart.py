"""UniChart (ahmed-masry/unichart-base-960, Donut, GPL-3.0): chart -> table model trained on charts.

Prompt "<extract_data_table> <s_answer>" from the model card. Output is DePlot style
("Year | Sales & 2019 | 10 & ..." with "&" between rows), turned into lines here.
"""
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

PROMPT = "<extract_data_table> <s_answer>"


class UniChartReader:
    def __init__(self, model="ahmed-masry/unichart-base-960", device=None, max_new_tokens=512, **_):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DonutProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.name = model

    @torch.inference_mode()
    def read(self, image_paths, n=1, temperature=0.0, seed=0, **_):
        torch.manual_seed(seed)
        tok = self.processor.tokenizer
        prompt_ids = tok(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        results = []
        for p in image_paths:
            pixels = self.processor(Image.open(p).convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
            sample = n > 1 or temperature > 0
            out = self.model.generate(
                pixels, decoder_input_ids=prompt_ids, max_new_tokens=self.max_new_tokens,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True,
                bad_words_ids=[[tok.unk_token_id]], num_beams=1 if sample else 4,
                do_sample=sample, temperature=temperature if sample else None, num_return_sequences=n)
            texts = []
            for seq in tok.batch_decode(out):
                seq = seq.replace(tok.eos_token, "").replace(tok.pad_token, "")
                seq = seq.split("<s_answer>")[-1].strip()
                texts.append(seq.replace(" & ", "\n").replace("&", "\n"))
            results.append(texts)
        return results
