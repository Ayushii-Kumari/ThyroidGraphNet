import os
import json
import shutil

import torch
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download

from config import Config
from utils.preprocessing import preprocess_single_image
from network import ThyroidNet


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Thyroid Nodule AI",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .prediction-card {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #1a1f2e 0%,
            #232838 100%
        );
        border: 1px solid #2d3348;
        margin-bottom: 1rem;
    }

    .badge-malignant,
    .badge-benign {
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }

    .badge-malignant {
        background-color: #ff4b4b;
    }

    .badge-benign {
        background-color: #21c55d;
    }

    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .bar-bg {
        background-color: #2d3348;
        border-radius: 8px;
        height: 22px;
        width: 100%;
        overflow: hidden;
    }

    .bar-fill-mal {
        background-color: #ff4b4b;
        height: 100%;
    }

    .bar-fill-ben {
        background-color: #21c55d;
        height: 100%;
    }

    .info-box {
        padding: 1rem;
        border-radius: 12px;
        background-color: #1a1f2e;
        border: 1px solid #2d3348;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_REPO_ID = "AyushiiKumarii/thyroidnet"

MODEL_FILENAME = "thyroidnet_best.pth"
SUPPORT_BANK_FILENAME = "support_bank.pt"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    cfg = Config()

    device = torch.device(cfg.DEVICE)

    model_dir = cfg.MODEL_DIR

    model_path = os.path.join(
        model_dir,
        MODEL_FILENAME
    )

    support_bank_path = os.path.join(
        model_dir,
        SUPPORT_BANK_FILENAME
    )

    print("=" * 60)
    print("MODEL LOADING")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Model directory: {model_dir}")
    print(f"Model path: {model_path}")
    print(f"Support bank path: {support_bank_path}")
    print(f"Hugging Face repository: {HF_REPO_ID}")

    # --------------------------------------------------------
    # Create models directory
    # --------------------------------------------------------

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # ========================================================
    # DOWNLOAD MODEL FROM HUGGING FACE
    # ========================================================

    try:

        # ----------------------------------------------------
        # Main model
        # ----------------------------------------------------

        if not os.path.exists(model_path):

            print(
                "thyroidnet_best.pth not found locally."
            )

            print(
                "Downloading thyroidnet_best.pth "
                "from Hugging Face..."
            )

            downloaded_model = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME
            )

            shutil.copy2(
                downloaded_model,
                model_path
            )

            print(
                "thyroidnet_best.pth downloaded successfully."
            )

        else:

            print(
                "thyroidnet_best.pth already exists locally."
            )

        # ----------------------------------------------------
        # Support bank
        # ----------------------------------------------------

        if not os.path.exists(support_bank_path):

            print(
                "support_bank.pt not found locally."
            )

            print(
                "Downloading support_bank.pt "
                "from Hugging Face..."
            )

            downloaded_support_bank = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=SUPPORT_BANK_FILENAME
            )

            shutil.copy2(
                downloaded_support_bank,
                support_bank_path
            )

            print(
                "support_bank.pt downloaded successfully."
            )

        else:

            print(
                "support_bank.pt already exists locally."
            )

    except Exception as e:

        error_message = (
            "Failed to download model files from "
            f"Hugging Face repository "
            f"'{HF_REPO_ID}'.\n\n"
            f"Error: {e}"
        )

        print(error_message)

        return (
            None,
            None,
            cfg,
            device,
            error_message
        )

    # ========================================================
    # VERIFY MODEL FILES
    # ========================================================

    print(
        f"Model exists: "
        f"{os.path.exists(model_path)}"
    )

    print(
        f"Support bank exists: "
        f"{os.path.exists(support_bank_path)}"
    )

    if not os.path.exists(model_path):

        return (
            None,
            None,
            cfg,
            device,
            f"Model not found: {model_path}"
        )

    if not os.path.exists(support_bank_path):

        return (
            None,
            None,
            cfg,
            device,
            f"Support bank not found: {support_bank_path}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        print(
            "Creating ThyroidNet model..."
        )

        model = ThyroidNet(
            cfg
        ).to(device)

        print(
            "Loading model checkpoint..."
        )

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint
        )

        model.eval()

        print(
            "Loading support bank..."
        )

        bank = torch.load(
            support_bank_path,
            map_location=device
        )

        # ----------------------------------------------------
        # Validate support bank
        # ----------------------------------------------------

        if isinstance(bank, dict):

            if "features" not in bank:

                return (
                    None,
                    None,
                    cfg,
                    device,
                    "support_bank.pt does not contain "
                    "the expected 'features' key."
                )

            support_feats = bank[
                "features"
            ].to(device)

        else:

            support_feats = bank.to(device)

        # ----------------------------------------------------
        # Successful loading
        # ----------------------------------------------------

        print(
            "Model loaded successfully."
        )

        print(
            f"Support bank shape: "
            f"{support_feats.shape}"
        )

        print("=" * 60)

        return (
            model,
            support_feats,
            cfg,
            device,
            None
        )

    except Exception as e:

        print(
            f"MODEL LOADING ERROR: {e}"
        )

        return (
            None,
            None,
            cfg,
            device,
            str(e)
        )


# ============================================================
# LOAD MODEL
# ============================================================

(
    model,
    support_feats,
    cfg,
    device,
    model_error
) = load_model()


# ============================================================
# UNCERTAINTY LABEL
# ============================================================

def uncertainty_label(u):

    if u < 0.15:

        return (
            "LOW UNCERTAINTY",
            "#21c55d"
        )

    elif u < 0.35:

        return (
            "MODERATE UNCERTAINTY",
            "#f5a623"
        )

    else:

        return (
            "HIGH UNCERTAINTY",
            "#ff4b4b"
        )


# ============================================================
# PREDICTION CONFIDENCE
# ============================================================

def prediction_confidence(probabilities):

    predicted_index = probabilities.argmax()

    return (
        float(
            probabilities[predicted_index]
        ) * 100
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🩺 Thyroid Nodule Classification"
)

st.caption(
    "Uncertainty-Guided Hybrid CNN–Transformer "
    "with Adaptive Graph Reasoning"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About this system"
    )

    st.markdown(
        """
        **Architecture**

        - CNN branch: ConvNeXt V2-Tiny
        - Transformer branch: Swin V2-Tiny
        - Fusion: Uncertainty-Guided Adaptive Gated Fusion
        - Graph: Uncertainty-Adaptive k-NN + GATv2
        - Uncertainty: MC Dropout + Predictive Entropy
        """
    )

    st.divider()

    st.markdown(
        """
        **Dataset**

        - Training: TN5000
        - Evaluation: TN5000 Test Set
        """
    )

    st.divider()

    st.caption(
        "Research/educational prototype — "
        "not a diagnostic device."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🔍 Diagnosis",
        "📊 Model Performance (TN5000 Test)"
    ]
)


# ============================================================
# TAB 1 — DIAGNOSIS
# ============================================================

with tab1:

    # --------------------------------------------------------
    # MODEL ERROR
    # --------------------------------------------------------

    if model is None:

        st.error(
            "The trained model could not be loaded."
        )

        if model_error:

            st.code(
                model_error
            )

        st.info(
            "The application automatically downloads "
            "the trained model from Hugging Face."
        )

        st.markdown(
            f"""
            **Hugging Face repository**

            `{HF_REPO_ID}`
            """
        )

        st.code(
            """
thyroidnet_best.pth
support_bank.pt
            """
        )

    # --------------------------------------------------------
    # MODEL SUCCESSFULLY LOADED
    # --------------------------------------------------------

    else:

        st.success(
            f"Model loaded successfully • Device: {device}"
        )

        uploaded = st.file_uploader(
            "Upload a thyroid ultrasound image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp"
            ]
        )

        # ----------------------------------------------------
        # IMAGE UPLOADED
        # ----------------------------------------------------

        if uploaded is not None:

            image = Image.open(
                uploaded
            ).convert("RGB")

            original_width, original_height = image.size

            col1, col2 = st.columns(
                [1, 1.2]
            )

            # =================================================
            # IMAGE
            # =================================================

            with col1:

                st.image(
                    image,
                    caption="Uploaded ultrasound image",
                    width="stretch"
                )

                st.caption(
                    f"Image size: "
                    f"{original_width} × "
                    f"{original_height}"
                )

            # =================================================
            # INFERENCE
            # =================================================

            with st.spinner(
                "Running model inference..."
            ):

                try:

                    tensor = preprocess_single_image(
                        image,
                        cfg.IMAGE_SIZE
                    ).to(device)

                    with torch.no_grad():

                        result = model.forward_inference(
                            tensor,
                            support_feats
                        )

                    probabilities = torch.softmax(
                        result["logits"],
                        dim=-1
                    ).squeeze(0).cpu().numpy()

                except Exception as e:

                    st.error(
                        "Model inference failed."
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()

            # =================================================
            # PROBABILITIES
            # =================================================

            benign_probability = (
                float(
                    probabilities[0]
                ) * 100
            )

            malignant_probability = (
                float(
                    probabilities[1]
                ) * 100
            )

            predicted_index = int(
                probabilities.argmax()
            )

            # =================================================
            # PREDICTION
            # =================================================

            if predicted_index == 1:

                prediction = "MALIGNANT"

                badge_class = (
                    "badge-malignant"
                )

            else:

                prediction = "BENIGN"

                badge_class = (
                    "badge-benign"
                )

            # =================================================
            # CONFIDENCE
            # =================================================

            classification_confidence = (
                prediction_confidence(
                    probabilities
                )
            )

            # =================================================
            # UNCERTAINTY
            # =================================================

            u1 = float(
                result["u1"]
            )

            u2 = float(
                result["u2"]
            )

            uncertainty_text, uncertainty_color = (
                uncertainty_label(
                    u2
                )
            )

            # =================================================
            # RESULT COLUMN
            # =================================================

            with col2:

                st.markdown(
                    '<div class="prediction-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <h3>
                        Prediction&nbsp;&nbsp;
                        <span class="{badge_class}">
                            {prediction}
                        </span>
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # ------------------------------------------------
                # MALIGNANT PROBABILITY
                # ------------------------------------------------

                st.markdown(
                    f"**Malignant — "
                    f"{malignant_probability:.2f}%**"
                )

                st.markdown(
                    f"""
                    <div class="bar-bg">
                        <div
                            class="bar-fill-mal"
                            style="width:{malignant_probability}%">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # ------------------------------------------------
                # BENIGN PROBABILITY
                # ------------------------------------------------

                st.markdown(
                    f"**Benign — "
                    f"{benign_probability:.2f}%**"
                )

                st.markdown(
                    f"""
                    <div class="bar-bg">
                        <div
                            class="bar-fill-ben"
                            style="width:{benign_probability}%">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                # =================================================
                # METRICS
                # =================================================

                m1, m2, m3 = st.columns(3)

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                m1.markdown(
                    f"""
                    <div class="metric-box">
                        <small>
                            Prediction Confidence
                        </small>
                        <br>
                        <b>
                            {classification_confidence:.2f}%
                        </b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # UNCERTAINTY
                # ------------------------------------------------

                m2.markdown(
                    f"""
                    <div class="metric-box">
                        <small>
                            Uncertainty (U₂)
                        </small>
                        <br>
                        <b style="color:{uncertainty_color}">
                            {u2:.4f}
                        </b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # GRAPH
                # ------------------------------------------------

                graph_type = str(
                    result["graph_type"]
                ).capitalize()

                k_value = int(
                    result["k"]
                )

                m3.markdown(
                    f"""
                    <div class="metric-box">
                        <small>
                            Graph
                        </small>
                        <br>
                        <b>
                            {graph_type}
                        </b>
                        <br>
                        k = {k_value}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # =================================================
                # UNCERTAINTY MESSAGE
                # =================================================

                st.info(
                    f"Model uncertainty: "
                    f"**{uncertainty_text}**"
                )

                # =================================================
                # ADVANCED DETAILS
                # =================================================

                with st.expander(
                    "Advanced model details"
                ):

                    st.write(
                        f"**Prediction:** "
                        f"{prediction}"
                    )

                    st.write(
                        f"**Prediction confidence:** "
                        f"{classification_confidence:.4f}%"
                    )

                    st.write(
                        f"**CNN–Transformer fusion "
                        f"uncertainty (U₁):** "
                        f"{u1:.6f}"
                    )

                    st.write(
                        f"**Graph uncertainty (U₂):** "
                        f"{u2:.6f}"
                    )

                    st.write(
                        f"**Graph type:** "
                        f"{graph_type}"
                    )

                    st.write(
                        f"**Graph neighbours (k):** "
                        f"{k_value}"
                    )

                    st.write(
                        f"**Input tensor shape:** "
                        f"{tuple(tensor.shape)}"
                    )

                    st.write(
                        f"**Input tensor device:** "
                        f"{tensor.device}"
                    )

                    st.write(
                        f"**Input tensor dtype:** "
                        f"{tensor.dtype}"
                    )

                    st.write(
                        f"**Tensor minimum:** "
                        f"{tensor.min().item():.4f}"
                    )

                    st.write(
                        f"**Tensor maximum:** "
                        f"{tensor.max().item():.4f}"
                    )

                    st.write(
                        f"**Tensor mean:** "
                        f"{tensor.mean().item():.4f}"
                    )

                    st.write(
                        f"**Tensor standard deviation:** "
                        f"{tensor.std().item():.4f}"
                    )

                    st.write(
                        "**Raw logits:**"
                    )

                    st.code(
                        str(
                            result["logits"]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    )


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================

with tab2:

    metrics_path = os.path.join(
        cfg.RESULTS_DIR,
        "metrics.json"
    )

    # --------------------------------------------------------
    # METRICS NOT FOUND
    # --------------------------------------------------------

    if not os.path.exists(
        metrics_path
    ):

        st.warning(
            "TN5000 evaluation results were not found."
        )

        st.code(
            metrics_path
        )

    # --------------------------------------------------------
    # METRICS FOUND
    # --------------------------------------------------------

    else:

        try:

            with open(
                metrics_path,
                "r"
            ) as f:

                metrics = json.load(f)

            st.subheader(
                "Test Performance on TN5000"
            )

            # =================================================
            # PRIMARY METRICS
            # =================================================

            cols = st.columns(4)

            metric_keys = [
                "accuracy",
                "precision",
                "sensitivity",
                "specificity"
            ]

            metric_labels = [
                "Accuracy",
                "Precision",
                "Sensitivity",
                "Specificity"
            ]

            for col, key, label in zip(
                cols,
                metric_keys,
                metric_labels
            ):

                if key in metrics:

                    col.metric(
                        label,
                        f"{metrics[key] * 100:.2f}%"
                    )

            # =================================================
            # SECONDARY METRICS
            # =================================================

            cols2 = st.columns(3)

            secondary_keys = [
                "f1_score",
                "roc_auc",
                "pr_auc"
            ]

            secondary_labels = [
                "F1-Score",
                "ROC-AUC",
                "PR-AUC"
            ]

            for col, key, label in zip(
                cols2,
                secondary_keys,
                secondary_labels
            ):

                if key in metrics:

                    if key == "f1_score":

                        value = (
                            f"{metrics[key] * 100:.2f}%"
                        )

                    else:

                        value = (
                            f"{metrics[key]:.4f}"
                        )

                    col.metric(
                        label,
                        value
                    )

            # =================================================
            # SAMPLE COUNT
            # =================================================

            if "n_samples" in metrics:

                st.caption(
                    f"Test samples: "
                    f"{metrics['n_samples']}"
                )

            st.divider()

            # =================================================
            # VISUALIZATIONS
            # =================================================

            st.subheader(
                "Evaluation Visualizations"
            )

            img_files = {

                "Confusion Matrix":
                    "confusion_matrix.png",

                "ROC Curve":
                    "roc_curve.png",

                "Precision-Recall Curve":
                    "pr_curve.png",

                "Calibration Curve":
                    "calibration.png"
            }

            img_cols = st.columns(2)

            for i, (
                title,
                filename
            ) in enumerate(
                img_files.items()
            ):

                file_path = os.path.join(
                    cfg.RESULTS_DIR,
                    filename
                )

                with img_cols[i % 2]:

                    st.markdown(
                        f"### {title}"
                    )

                    if os.path.exists(
                        file_path
                    ):

                        st.image(
                            file_path,
                            width="stretch"
                        )

                    else:

                        st.caption(
                            "Not generated."
                        )

        except Exception as e:

            st.error(
                "Could not read the evaluation results."
            )

            st.code(
                str(e)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Thyroid Nodule AI • "
    "Uncertainty-Guided Hybrid CNN–Transformer "
    "with Adaptive Graph Reasoning"
)

st.caption(
    "Research/educational prototype — "
    "not a diagnostic device."
)