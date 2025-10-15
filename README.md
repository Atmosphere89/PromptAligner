# 🌐 PromptAligner: Adaptive Prompt Refinement for Image Generation

PromptAligner is an experimental system designed to continuously improve the alignment between user intent and AI-generated images.
It optimizes prompts dynamically through semantic, aesthetic, and structural evaluation — while learning from user feedback over time.

---

## 🚀 Overview

Modern text-to-image models (e.g., Stable Diffusion, SDXL, DALL·E 3, FLUX) can generate high-quality images,
but often fail to precisely match the intended meaning or emotional nuance behind the user’s prompt.
PromptAligner introduces a feedback-driven optimization loop that bridges this gap.

---

## 🧠 Core Concept

PromptAligner evaluates every image along three harmony dimensions and iteratively refines the prompt to maximize overall consistency.

| Symbol | Name | Description |
|:--:|:--|:--|
| v_c | Content Consistency | Measures semantic overlap between prompt and generated image (via CLIP/BLIP embeddings). |
| v_a | Aesthetic Alignment | Scores visual style and aesthetic quality using pretrained aesthetic models. |
| v_s | Structural Similarity | Compares layout or composition using segmentation and keypoint matching. |

The Harmony Index (H) aggregates these with user-defined weights (α, β, γ):

`H = (v_c^α × v_a^β × v_s^γ)^(1/(α+β+γ))`

---

## 🔁 Adaptive Prompt Loop

```
[User Prompt] → [Image Generator] → [Generated Image]
        ↓                             ↑
 [Evaluator: Harmony Index] ← [Adaptive Intent Rewriter]
```

Each cycle:
1. Generate image using the current prompt.
2. Evaluate Harmony Index (H).
3. Use the Adaptive Intent Rewriter (LLM-based) to modify the prompt based on semantic differences.
4. Repeat until H exceeds a stability threshold.

---

## 💬 Feedback Consistency Module (FCM)

A lightweight self-adjusting mechanism that interprets user reactions (likes, retries, improvements)
and measures the semantic gap between user expectation and model output.

### 🎯 Purpose

Quantify how far the output deviated from the user’s true intent — and feed that insight back into prompt evolution.

### ⚙️ How It Works

1. Collects user events (retry, dislike, improve) and natural-language feedback.
2. Extracts image caption (via BLIP or CLIP Interrogator).
3. Computes similarity between prompt, feedback, and caption.
4. Calculates Consistency Deviation Score (CDS):

`CDS = 1 - (1/3) * (PFS + PIA + FIA)`

| Term | Meaning |
|------|----------|
| PFS | Prompt–Feedback semantic similarity |
| PIA | Prompt–Image alignment |
| FIA | Feedback–Image alignment |

→ High CDS = large mismatch.  
→ CDS is used to automatically trigger prompt rewriting or suggest re-generation.

### Example Implementation

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calc_cds(prompt, feedback, caption):
    p_emb = model.encode(prompt, convert_to_tensor=True)
    f_emb = model.encode(feedback, convert_to_tensor=True)
    c_emb = model.encode(caption, convert_to_tensor=True)

    pfs = util.cos_sim(p_emb, f_emb).item()
    pia = util.cos_sim(p_emb, c_emb).item()
    fia = util.cos_sim(f_emb, c_emb).item()

    cds = 1 - (pfs + pia + fia) / 3
    return max(0, min(1, cds))  # normalized 0–1
```

---

## 📁 Project Structure

```
PromptAligner/
├── app.py                     # Gradio or Streamlit UI
├── core/
│   ├── generator.py           # Calls SDXL / DALL·E / Flux models
│   ├── evaluator.py           # Harmony Index computation
│   ├── rewriter.py            # Adaptive Intent Rewriter (LLM)
│   ├── feedback_module.py     # Feedback Consistency Module (FCM)
│   └── utils.py
├── models/
│   ├── clip_model/
│   ├── aesthetic_model/
│   └── structure_model/
├── requirements.txt
└── README.md
```

---

## 🔬 Key Features

✅ Dynamic Prompt Refinement – Iteratively adjusts prompts to improve semantic fidelity.  
✅ User Feedback Learning – Captures human reactions to fine-tune system understanding.  
✅ Multi-Axis Evaluation – Balances meaning, beauty, and structure for realistic intent alignment.  
✅ No Training Required – All optimization is inference-based; works on-the-fly.  
✅ Open Modular Design – Each component (Generator, Evaluator, FCM) can be replaced or extended.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourname/PromptAligner.git
cd PromptAligner
pip install -r requirements.txt
```

Run the app:
```bash
python app.py
```

---

## 💡 Future Directions

- **Voice or sketch-based intent capture**  
  Allow users to express “what they meant” via multimodal input.
- **Personal preference learning**  
  Adapt weighting (α, β, γ) per user profile.
- **Transparency Dashboard**  
  Visualize each iteration’s evaluation and feedback scores in real time.

---

## 🪄 License

Apache License 2.0 © 2025 your-name  
Please cite the repository if used in research or derivative works.

---

## 🧭 Summary

PromptAligner is an independent, open framework —  
not tied to any prior theoretical model —  
focusing purely on practical, feedback-driven prompt evolution.  
It brings human perception into the image generation loop while remaining lightweight and interpretable.
