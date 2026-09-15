from math import floor

def allocate(n, ratios):          # ratios: {"A": 0.1, "B": 0.3, "C": 0.6}
    total = sum(ratios.values())  # marche aussi avec des poids bruts (1,3,6)
    exact = {k: n * w / total for k, w in ratios.items()}
    counts = {k: floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[:n - sum(counts.values())]:
        counts[k] += 1
    return counts