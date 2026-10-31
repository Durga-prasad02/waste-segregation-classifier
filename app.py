"""
Smart-Bin Sorting Assistant - Streamlit Web UI
Waste Segregation Image Classifier & Smart Waste Segregation Assistant
Candidate: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri
Domain: Deep Learning & Computer Vision
"""

import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Import project modules
from model import WASTE_CLASSES, CATEGORY_METADATA
from predict import predict_waste_image

# Page configuration
st.set_page_config(
    page_title="Waste Segregation Image Classifier & Smart Waste Segregation Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished presentation & smart bin visualization
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b5e20;
        text-align: center;
        margin-top: 0.2rem;
        margin-bottom: 0.4rem;
        line-height: 1.25;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #37474f;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .bin-card {
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-top: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .instruction-box {
        background-color: #fffde7;
        border-left: 4px solid #fbc02d;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for smart-bin statistics
if "scanned_count" not in st.session_state:
    st.session_state.scanned_count = 0
if "recyclable_count" not in st.session_state:
    st.session_state.recyclable_count = 0
if "history_log" not in st.session_state:
    st.session_state.history_log = []


def render_sidebar():
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=600&auto=format&fit=crop&q=60", width="stretch")
        st.title("🌱 Smart-Bin Hub")
        st.markdown("**Autonomous Waste Segregation Prototype**")
        st.divider()

        # Project Details
        st.markdown("### 🎓 Academic Credentials")
        st.markdown("""
        - **Candidate:** AGGIBADALA VENKATA DURGA NAGA PRASAD
        - **Roll No:** `23ME1A5402`
        - **Mentor:** Ms. G. Hema Sri
        - **Domain:** Deep Learning & Computer Vision
        """)
        st.divider()

        # Smart Bin Cumulative Statistics
        st.markdown("### 📊 Smart-Bin Session Stats")
        total = st.session_state.scanned_count
        recyclable = st.session_state.recyclable_count
        recyc_rate = (recyclable / total * 100) if total > 0 else 0.0

        col1, col2 = st.columns(2)
        col1.metric("Scanned", f"{total}")
        col2.metric("Recyclable", f"{recyclable}")
        st.progress(recyc_rate / 100.0)
        st.caption(f"Diversion Efficiency: **{recyc_rate:.1f}%**")

        st.divider()
        st.markdown("### ⚙️ Model Settings")
        selected_model_path = st.selectbox(
            "Model Checkpoint",
            options=["waste_classifier_model.pth"],
            index=0
        )
        use_gradcam = st.toggle("Enable Grad-CAM Heatmap", value=True)
        
        if st.button("🗑️ Reset Session Counters"):
            st.session_state.scanned_count = 0
            st.session_state.recyclable_count = 0
            st.session_state.history_log = []
            st.rerun()

        st.caption("Powered by PyTorch CNN & Streamlit")
    return selected_model_path, use_gradcam


def display_prediction_results(image: Image.Image, model_path: str, use_gradcam: bool):
    with st.spinner("🔍 CNN Feature Extraction & Classification in progress..."):
        time.sleep(0.3)
        results = predict_waste_image(image, model_path=model_path, compute_gradcam=use_gradcam)

    predicted_class = results["predicted_class"]
    confidence = results["confidence"]
    meta = results["metadata"]
    is_recyclable = meta["category_type"] == "Recyclable"

    # Update session counters
    st.session_state.scanned_count += 1
    if is_recyclable:
        st.session_state.recyclable_count += 1
    st.session_state.history_log.append({
        "Time": time.strftime("%H:%M:%S"),
        "Category": meta["label"],
        "Type": meta["category_type"],
        "Confidence": f"{confidence * 100:.1f}%"
    })

    # Top Banner: Bin Recommendation
    bin_bg = meta["bin_color"]
    st.markdown(f"""
    <div class="bin-card" style="background-color: {bin_bg};">
        <h2 style="margin: 0; color: white;">{meta['icon']} Classified as: {meta['label']}</h2>
        <h4 style="margin: 5px 0 0 0; color: #f0f0f0;">Target Destination: <b>{meta['bin_name']}</b></h4>
        <p style="margin: 8px 0 0 0; font-size: 1.1rem;">Recyclability Status: <b>{'✅ RECYCLABLE' if is_recyclable else '⚠️ NON-RECYCLABLE / COMPOST'}</b> (Confidence: <b>{confidence * 100:.2f}%</b>)</p>
    </div>
    """, unsafe_allow_html=True)

    # 2 Column layout: Visual Inspection vs Actionable Instructions
    col_img, col_info = st.columns([1.1, 1.2])

    with col_img:
        st.subheader("🖼️ Vision & Model Interpretability")
        if use_gradcam and results.get("gradcam_overlay") is not None:
            tab_orig, tab_cam, tab_heat = st.tabs(["Original Image", "Grad-CAM Activation", "Raw Heatmap"])
            with tab_orig:
                st.image(image, caption="Input Waste Item", width="stretch")
            with tab_cam:
                st.image(results["gradcam_overlay"], caption="Grad-CAM Focus Overlay (Conv4 Features)", width="stretch")
                st.caption("🔴 Red/Yellow highlights denote features that strongly influenced the CNN decision.")
            with tab_heat:
                st.image(results["heatmap"], caption="Salience Activation Map", width="stretch")
        else:
            st.image(image, caption="Analyzed Waste Item", width="stretch")

    with col_info:
        st.subheader("📋 Smart-Bin Disposal Guidelines")
        st.markdown(f"""
        <div class="instruction-box">
            <b>Disposal Action:</b> {meta['disposal_instructions']}
        </div>
        """, unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="metric-box">
                <b>⏳ Decomposition Time</b><br>
                <span style="font-size: 1.1rem; color: #d32f2f; font-weight: bold;">{meta['decomposition_time']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-box">
                <b>🌍 Environmental Impact</b><br>
                <span style="font-size: 0.85rem; color: #2e7d32;">{meta['carbon_saving']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"💡 **Practical Tips:** {meta['tips']}")

        # Probability Distribution Bar Chart
        st.subheader("📊 Category Probabilities")
        prob_df = pd.DataFrame([
            {"Category": CATEGORY_METADATA[k]["label"], "Probability (%)": v * 100}
            for k, v in results["probabilities"].items()
        ]).sort_values(by="Probability (%)", ascending=True)
        
        st.bar_chart(prob_df.set_index("Category"))


def main():
    model_path, use_gradcam = render_sidebar()

    # Header centered at the top of the app
    st.markdown('<div class="main-title">♻️ Waste Segregation Image Classifier & Smart Waste Segregation Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Design and Development of a Working Prototype for Waste Segregation Image Classifier Using a CNN Trained on Waste-Category Images</div>', unsafe_allow_html=True)

    with st.expander("📖 Project Abstract & Research Summary", expanded=False):
        st.write("""
        **Abstract:** This work presents a CNN trained on waste-category images to sort waste images into recyclable categories, 
        addressing the problem of waste segregation image classifier through a data-driven approach. 
        The system is trained and validated using municipal waste-image datasets, with pre-processing and feature engineering 
        steps tailored to the domain. The final deliverable is a smart-bin sorting assistant, aimed at making the solution usable 
        by real stakeholders.
        """)

    # Main Interactive Tabs
    tab_upload, tab_camera, tab_samples = st.tabs([
        "📤 Image Upload", 
        "📷 Smart-Bin Camera", 
        "🧪 Demonstration Gallery"
    ])

    with tab_upload:
        st.write("Upload an image of a waste item (bottle, paper, can, packaging, organic scrap) to test sorting:")
        uploaded_file = st.file_uploader("Select Waste Image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            display_prediction_results(image, model_path, use_gradcam)
        else:
            st.info("👆 Please upload an image or select the **'Demonstration Gallery'** tab for instant sample testing.")

    with tab_camera:
        st.write("Point your laptop or smart-bin connected webcam at the waste item:")
        cam_image = st.camera_input("Capture Waste Item for Real-time Sorting")
        if cam_image is not None:
            image = Image.open(cam_image)
            display_prediction_results(image, model_path, use_gradcam)

    with tab_samples:
        st.write("Select a pre-loaded municipal waste benchmark sample for instant evaluation:")
        samples_dir = Path("samples")
        if not samples_dir.exists():
            from dataset_downloader import setup_dataset_structure
            setup_dataset_structure()

        sample_cols = st.columns(6)
        selected_sample_img = None

        for idx, cls in enumerate(WASTE_CLASSES):
            sample_file = samples_dir / f"demo_{cls}.jpg"
            if sample_file.exists():
                with sample_cols[idx]:
                    img = Image.open(sample_file)
                    st.image(img, caption=CATEGORY_METADATA[cls]["label"], width="stretch")
                    if st.button(f"Test {cls.capitalize()}", key=f"btn_{cls}"):
                        selected_sample_img = img

        if selected_sample_img is not None:
            st.divider()
            display_prediction_results(selected_sample_img, model_path, use_gradcam)

    if st.session_state.history_log:
        st.divider()
        with st.expander("📜 Recent Smart-Bin Inferences Session Log", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.history_log), width="stretch")


if __name__ == "__main__":
    main()
