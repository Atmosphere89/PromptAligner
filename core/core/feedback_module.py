# ===========================
# core/feedback_module.py — Feedback Consistency Module (FCM)
# ===========================

from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
# You can replace this with any other similarity model later
model = SentenceTransformer('all-MiniLM-L6-v2')

def calc_cds(prompt: str, feedback: str, caption: str) -> float:
    """
    Calculate the Consistency Deviation Score (CDS) based on semantic similarities.

    Parameters
    ----------
    prompt : str
        The original user prompt.
    feedback : str
        User feedback (e.g., improvement request or correction).
    caption : str
        Description generated from the image or model output.

    Returns
    -------
    float
        Consistency deviation score between 0 and 1.
        (Higher = larger mismatch between intent and result)
    """
    # Encode the text inputs
    p_emb = model.encode(prompt, convert_to_tensor=True)
    f_emb = model.encode(feedback, convert_to_tensor=True)
    c_emb = model.encode(caption, convert_to_tensor=True)

    # Compute pairwise cosine similarities
    pfs = util.cos_sim(p_emb, f_emb).item()
    pia = util.cos_sim(p_emb, c_emb).item()
    fia = util.cos_sim(f_emb, c_emb).item()

    # Combine into a single deviation score
    cds = 1 - (pfs + pia + fia) / 3
    return max(0, min(1, cds))  # Normalize to 0–1
