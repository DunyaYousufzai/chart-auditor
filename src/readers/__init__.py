"""Chart readers: picture -> table text.

Every reader has the same method:
    read(image_paths, n, temperature) -> list of lists of strings
    (for each image, n texts; the texts are parsed later by tables.parse_model_output)

get_reader(name, **kw) builds one by name, so the scripts do not import heavy
libraries (torch, vllm) unless that reader is used.
"""


def get_reader(name, **kw):
    if name == "deplot":
        from readers.deplot import DePlotReader
        return DePlotReader(**kw)
    if name in ("qwen", "qwen-vllm"):
        from readers.qwen_vl import QwenVLLMReader
        return QwenVLLMReader(**kw)
    if name == "qwen-hf":
        from readers.qwen_vl import QwenHFReader
        return QwenHFReader(**kw)
    if name == "unichart":
        from readers.unichart import UniChartReader
        return UniChartReader(**kw)
    if name == "chartgemma":
        from readers.chartgemma import ChartGemmaReader
        return ChartGemmaReader(**kw)
    if name == "claude":
        from readers.api_claude import ClaudeReader
        return ClaudeReader(**kw)
    if name == "openai":
        from readers.api_openai import OpenAIReader
        return OpenAIReader(**kw)
    if name == "gemini":
        from readers.api_gemini import GeminiReader
        return GeminiReader(**kw)
    if name == "simulated":
        from readers.simulated import SimulatedReader
        return SimulatedReader(**kw)
    raise ValueError(f"unknown reader {name}")
