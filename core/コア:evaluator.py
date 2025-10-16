# ===========================
# core/evaluator.py — Harmony Index computation
# ===========================

def compute_harmony_index(v_c: float, v_a: float, v_s: float, weights=(1, 1, 1)) -> float:
    """
    Compute the Harmony Index (H) using a weighted geometric mean.

    Parameters
    ----------
    v_c : float
        Content Consistency (semantic similarity)
    v_a : float
        Aesthetic Alignment (visual style score)
    v_s : float
        Structural Similarity (composition matching)
    weights : tuple
        (alpha, beta, gamma) weight coefficients

    Returns
    -------
    float
        The harmony score (H) between 0 and 1.
    """
    alpha, beta, gamma = weights
    product = (v_c ** alpha) * (v_a ** beta) * (v_s ** gamma)
    exponent = 1 / (alpha + beta + gamma)
    return product ** exponent
