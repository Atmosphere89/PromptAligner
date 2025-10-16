# ===========================
# app.py — PromptAligner main interface
# ===========================

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import gradio as gr
from core.evaluator import compute_harmony_index
from core.feedback_module import calc_cds
from core.rewriter import refine_prompt


def generate_image(prompt):
    """
    Placeholder for image generation call.
    You can later connect this to Stable Diffusion, SDXL, or any other model.
    """
    return f"Generated image based on prompt: {prompt}"


def main_interface(prompt, feedback=None):
    """
    Main loop connecting the evaluator, feedback analyzer, and rewriter.
    """
    # Placeholder values for Harmony Index calculation
    v_c, v_a, v_s = 0.8, 0.7, 0.9
    harmony = compute_harmony_index(v_c, v_a, v_s)
    result_text = f"Harmony Index: {harmony:.3f}"

    # Optional feedback scoring
    if feedback:
        caption = "Placeholder image description"
        cds = calc_cds(prompt, feedback, caption)
        result_text += f" | Consistency Deviation Score: {cds:.3f}"

    # Prompt refinement step
    improved_prompt = refine_prompt(prompt)
    return result_text, improved_prompt


# Create a simple Gradio UI
iface = gr.Interface(
    fn=main_interface,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate"),
        gr.Textbox(label="Feedback (optional)", placeholder="Enter feedback or improvement request")
    ],
    outputs=[
        gr.Textbox(label="Evaluation"),
        gr.Textbox(label="Refined Prompt")
    ],
    title="PromptAligner Prototype",
    description="Adaptive prompt refinement with feedback-based evaluation"
)

if __name__ == "__main__":
    iface.launch()
