def compare(chart_values, table_values, tolerance=0.05):
    """Compare the numbers read from the chart with the numbers in the table.

    chart_values / table_values: dictionaries like {"A": 80, "B": 72}
    tolerance: the allowed gap, 0.05 means 5%


    c = the value read from the chart.
    t = the value in the table.
    error = |c - t| / |t| is the relative error. For B: |60 - 72| / 72 = 0.167.
    tolerance = 0.05 means we accept gaps up to 5%. Above that, we raise a flag. 
    This is the fixed tolerance. Later we replace it with our smart one, max(τ, k·s).
    Picture in mind: around each table value there is an invisible band of ±5%. If the chart bar ends inside the band, it is fine. If it ends outside, we flag it.
    Why check t != 0? Dividing by zero would crash the code. 
    If the table value is 0, we use the plain gap instead.

    """
    differences = []
    checked = 0

    for label, t in table_values.items():
        c = chart_values.get(label)
        if c is None:
            continue  # the chart has no value for this label
        checked += 1

        error = abs(c - t) / abs(t) if t != 0 else abs(c - t)
        if error > tolerance:
            differences.append(
                {"label": label, "chart": c, "table": t, "error": round(error, 4)}
            )

    if checked == 0:
        return {
            "verdict": "not_sure",
            "note": "No label of the chart matches a label of the table.",
            "differences": [],
        }

    if differences:
        return {
            "verdict": "mismatch",
            "note": f"{len(differences)} of {checked} values differ by more than {tolerance:.0%}. Please check by hand.",
            "differences": differences,
        }

    return {
        "verdict": "match",
        "note": f"All {checked} values agree within {tolerance:.0%}.",
        "differences": [],
    }