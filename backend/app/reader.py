def read_chart(filename, image_bytes, table):
    """FAKE chart reader. It does not look at the picture.

    It only exists so we can test the whole system.
    The real readers live in src/readers (DePlot, Qwen3-VL) and will replace this.
    Returns (chart_table, spread) like the real pipeline after combining N readings.
    """
    chart = table.copy()
    if "mismatch" in (filename or "").lower() and len(chart.rows) > 1:
        chart.values[1][0] = round(chart.values[1][0] * 5 / 6, 1)
    return chart, {}
