import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import gradio as gr
from app.predict import predict_interaction
from app.explain import generate_explanation
 
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
*, body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: #f4f6f9 !important;
    color: #1a1d23 !important;
}
 
footer, .built-with { display: none !important; }
 
/* ── Header ── */
#header {
    padding: 0 !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
}
 
/* ── All blocks ── */
.block, .gr-block, .gr-box, .gr-group,
.gradio-group, div.svelte-vt1mxs {
    background: #ffffff !important;
    border: 0.5px solid #dde1e7 !important;
    border-radius: 10px !important;
}
 
/* ── Inputs ── */
input[type=text], textarea, .scroll-hide {
    background: #ffffff !important;
    color: #1a1d23 !important;
    border: 0.5px solid #c8cdd6 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
input[type=text]:focus, textarea:focus {
    border-color: #4f8ef7 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important;
}
input[type=text]::placeholder, textarea::placeholder {
    color: #adb4bf !important;
}
 
/* ── Labels ── */
label > span, .label-wrap span {
    color: #6b7280 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
 
/* ── Button ── */
button.primary, #analyze-btn {
    background: #4f8ef7 !important;
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}
button.primary:hover { background: #3b7de8 !important; }
 
/* ── Output boxes ── */
.output-text textarea {
    background: #ffffff !important;
    color: #1a1d23 !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
    border: 0.5px solid #dde1e7 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
 
/* ── Examples ── */
.examples-table table {
    background: #ffffff !important;
    border: 0.5px solid #dde1e7 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
.examples-table td, .examples-table th {
    color: #6b7280 !important;
    font-size: 13px !important;
    border-color: #dde1e7 !important;
    padding: 10px 16px !important;
}
.examples-table tr:hover td {
    background: #f4f6f9 !important;
    color: #1a1d23 !important;
    cursor: pointer !important;
}
 
/* ── Markdown ── */
.prose h1, h1 {
    color: #1a1d23 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}
.prose p, p {
    color: #6b7280 !important;
    font-size: 13px !important;
}
"""
 
HEADER_HTML = """
<div style="
    background: #ffffff;
    border-bottom: 1px solid #dde1e7;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
    border-radius: 0;
">
    <div style="
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #4f8ef7;
    "></div>
    <span style="
        color: #1a1d23;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.3px;
    ">DrugSafe AI</span>
</div>
"""
 
STATS_HTML = """
<div style="
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 4px 0 8px 0;
">
    <div style="
        background: #ffffff;
        border: 0.5px solid #dde1e7;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    ">
        <div style="color: #1a1d23; font-size: 22px; font-weight: 700; line-height: 1.2">49,505</div>
        <div style="color: #6b7280; font-size: 11px; margin-top: 4px; letter-spacing: 0.3px">Drug pairs trained</div>
    </div>
    <div style="
        background: #ffffff;
        border: 0.5px solid #dde1e7;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    ">
        <div style="color: #1a1d23; font-size: 22px; font-weight: 700; line-height: 1.2">86%</div>
        <div style="color: #6b7280; font-size: 11px; margin-top: 4px; letter-spacing: 0.3px">Model accuracy</div>
    </div>
    <div style="
        background: #ffffff;
        border: 0.5px solid #dde1e7;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    ">
        <div style="color: #1a1d23; font-size: 22px; font-weight: 700; line-height: 1.2">3</div>
        <div style="color: #6b7280; font-size: 11px; margin-top: 4px; letter-spacing: 0.3px">Risk levels</div>
    </div>
</div>
"""
 
def run_prediction(drug_a, drug_b):
    if not drug_a.strip() or not drug_b.strip():
        return "—", "—", "", ""
 
    prediction = predict_interaction(drug_a.strip(), drug_b.strip())
 
    if 'error' in prediction:
        return f"⚠ {prediction['error']}", "—", "", ""
 
    risk   = prediction['risk_level']
    probs  = prediction['probabilities']
    shap   = prediction['top_shap_features']
 
    risk_icons = {
        'Minor':    '🟢  Minor',
        'Moderate': '🟡  Moderate',
        'Major':    '🔴  Major'
    }
    risk_display = risk_icons.get(risk, risk)
 
    prob_display = (
        f"  Minor       {probs['Minor']*100:.1f}%\n"
        f"  Moderate    {probs['Moderate']*100:.1f}%\n"
        f"  Major       {probs['Major']*100:.1f}%"
    )
 
    shap_lines = ""
    for i, f in enumerate(shap, 1):
        feat  = f['feature'].replace('A_', 'Drug A — ').replace('B_', 'Drug B — ').replace('_', ' ').title()
        shap_lines += (
            f"  {i}.  {feat}\n"
            f"       Value   :  {f['value']}\n"
            f"       Impact  :  {f['shap_score']}\n\n"
        )
 
    explanation = generate_explanation(prediction)
    disclaimer  = (
        "\n──────────────────────────────────────\n"
        "This information is intended for reference purposes only. "
        "Always consult your physician or a licensed healthcare "
        "professional before taking, combining, or adjusting any medications."
    )
 
    return risk_display, prob_display, shap_lines.strip(), explanation + disclaimer
 
 
with gr.Blocks(title="DrugSafe AI") as app:
 
    gr.HTML(HEADER_HTML)
 
    gr.Markdown("# Drug Interaction Risk Predictor")
    gr.Markdown("ML-powered prediction with explainable AI and clinical reasoning")
 
    gr.HTML(STATS_HTML)
 
    with gr.Row():
        drug_a_input = gr.Textbox(
            label="Drug A",
            placeholder="e.g. Carboplatin",
            scale=1
        )
        drug_b_input = gr.Textbox(
            label="Drug B",
            placeholder="e.g. Telbivudine",
            scale=1
        )
 
    analyze_btn = gr.Button(
        "Analyze Interaction",
        variant="primary",
        elem_id="analyze-btn"
    )
 
    with gr.Row():
        with gr.Column(scale=1):
            risk_output = gr.Textbox(
                label="Risk Level",
                interactive=False,
                lines=2,
                elem_classes=["output-text"]
            )
            prob_output = gr.Textbox(
                label="Confidence Score",
                interactive=False,
                lines=4,
                elem_classes=["output-text"]
            )
            gr.Examples(
                examples=[
                    ["Carboplatin",   "Telbivudine"],
                    ["Chloroquine",   "Paclitaxel"],
                    ["Metronidazole", "Warfarin"],
                    ["Disulfiram",    "Abacavir"],
                ],
                inputs=[drug_a_input, drug_b_input],
                label="Try an example",
            )
 
        with gr.Column(scale=1):
            shap_output = gr.Textbox(
                label="Top Contributing Features",
                interactive=False,
                lines=8,
                elem_classes=["output-text"]
            )
            explanation_output = gr.Textbox(
                label="Clinical Explanation",
                interactive=False,
                lines=9,
                elem_classes=["output-text"]
            )
 
    analyze_btn.click(
        fn=run_prediction,
        inputs=[drug_a_input, drug_b_input],
        outputs=[risk_output, prob_output, shap_output, explanation_output]
    )
 
if __name__ == '__main__':
    app.launch(css=CSS)