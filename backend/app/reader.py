def read_chart(filename, image_bytes, table_values):
    """FAKE chart reader. It does not look at the picture.

    It only exists so we can test the whole system.
    The real model (Qwen3-VL) will replace this function.
    """
    values = dict(table_values)

    if "mismatch" in (filename or "").lower():
        labels = list(values)
        if len(labels) > 1:
            values[labels[1]] = round(values[labels[1]] * 5 / 6, 1)

    return values