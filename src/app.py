
"""Premium dark-green Streamlit UI for Crop Disease Detection.

Run from project root:
    python -m streamlit run src/app.py

The app does NOT automatically determine whether an arbitrary photo is a leaf.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------- Paths / settings ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "mobilenetv2_cpu"
    / "color" / "best_model.keras"
)

CLASS_MAPPING_PATH = (
    PROJECT_ROOT / "models" / "mobilenetv2_cpu"
    / "color" / "class_mapping.json"
)

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_SIDE = 32
LOW_SCORE_WARNING = 0.60

ALLOWED_IMAGE_FORMATS = {
    "JPEG", "PNG", "BMP", "WEBP", "TIFF", "GIF"
}

EXTENSION_TO_FORMATS = {
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".png": {"PNG"},
    ".bmp": {"BMP"},
    ".webp": {"WEBP"},
    ".tif": {"TIFF"},
    ".tiff": {"TIFF"},
    ".gif": {"GIF"},
}

SRC_DIR = str(PROJECT_ROOT / "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from gradcam import make_gradcam, save_visualizations
except Exception as exc:
    make_gradcam = save_visualizations = None
    GRADCAM_IMPORT_ERROR = exc
else:
    GRADCAM_IMPORT_ERROR = None

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("crop_disease_app")

st.set_page_config(
    page_title="Crop Disease Intelligence",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------- Styling ---------------------------------

st.markdown(
    """
    <style>
    :root {
        --black: #070A07;
        --black-soft: #0D120E;
        --green: #1F8A4C;
        --white: #F5F8F5;
        --red: #D92D3E;
    }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"] {
        background: var(--black) !important;
        color: var(--white) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: var(--black-soft) !important;
        border-right: 1px solid var(--green);
    }

    [data-testid="stSidebar"] * {
        color: var(--white) !important;
    }

    h1, h2, h3, h4, p, label, li {
        color: var(--white);
    }

    h1, h2, h3 {
        letter-spacing: -0.035em;
    }

    p, li, label {
        line-height: 1.55;
    }

    a {
        color: var(--green) !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    /* HEADER */

    .brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: .4rem 0 1.1rem 0;
        border-bottom: 1px solid rgba(31,138,76,.55);
        margin-bottom: 1.6rem;
    }

    .brand-mark {
        display: flex;
        align-items: center;
        gap: .7rem;
        color: var(--white);
        font-weight: 850;
        letter-spacing: .13em;
        font-size: .86rem;
    }

    .brand-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: var(--green);
        color: var(--white);
        font-weight: 900;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        border: 1px solid var(--green);
        color: var(--white);
        background: rgba(31,138,76,.12);
        border-radius: 999px;
        padding: .42rem .78rem;
        font-size: .67rem;
        letter-spacing: .12em;
        font-weight: 800;
        white-space: nowrap;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
    }

    /* HERO */

    .hero {
        background:
            linear-gradient(
                125deg,
                rgba(31,138,76,.22),
                rgba(7,10,7,.12) 58%
            ),
            var(--black-soft);
        border: 1px solid rgba(31,138,76,.55);
        border-radius: 24px;
        padding: clamp(1.4rem, 3vw, 2.6rem);
        position: relative;
        overflow: hidden;
        margin-bottom: 1.3rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        right: -78px;
        top: -105px;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        border: 1px solid rgba(31,138,76,.35);
        box-shadow:
            0 0 0 26px rgba(31,138,76,.045),
            0 0 0 54px rgba(31,138,76,.035);
        pointer-events: none;
    }

    .eyebrow, .section-kicker {
        color: var(--green) !important;
        font-size: .7rem;
        font-weight: 900;
        letter-spacing: .19em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }

    .hero-title {
        color: var(--white);
        font-weight: 950;
        letter-spacing: -.065em;
        font-size: clamp(2.6rem, 5.5vw, 5rem);
        line-height: .93;
        margin: 0 0 1rem 0;
        text-transform: uppercase;
    }

    .hero-title span {
        color: var(--green);
    }

    .hero-copy {
        color: rgba(245,248,245,.82);
        font-size: 1rem;
        max-width: 690px;
    }

    .hero-side {
        background: rgba(7,10,7,.72);
        border: 1px solid rgba(31,138,76,.5);
        border-radius: 18px;
        padding: 1.1rem;
        position: relative;
        z-index: 1;
    }

    .hero-stat {
        padding: .72rem 0;
        border-bottom: 1px solid rgba(245,248,245,.12);
    }

    .hero-stat:last-child {
        border-bottom: 0;
    }

    .hero-stat b {
        display: block;
        color: var(--white);
        font-size: 1.18rem;
    }

    .hero-stat small {
        color: rgba(245,248,245,.64);
        letter-spacing: .12em;
        font-size: .64rem;
    }

    /* WORKFLOW */

    .step-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: .55rem;
        margin: 1.1rem 0 1.4rem 0;
    }

    .step-item {
        background: var(--black-soft);
        border: 1px solid rgba(31,138,76,.4);
        border-radius: 13px;
        padding: .72rem .8rem;
        display: flex;
        align-items: center;
        gap: .6rem;
        color: var(--white);
        font-weight: 750;
        font-size: .76rem;
    }

    .step-no {
        color: var(--green);
        font-size: .69rem;
        font-weight: 950;
    }

    /* PANELS */

    .panel {
        background: var(--black-soft);
        border: 1px solid rgba(31,138,76,.45);
        border-radius: 19px;
        padding: 1.15rem;
        height: 100%;
    }

    .panel-title {
        color: var(--white);
        font-size: 1rem;
        font-weight: 850;
        margin: 0 0 .35rem 0;
    }

    .muted {
        color: rgba(245,248,245,.67) !important;
        font-size: .86rem;
    }

    .upload-note {
        color: rgba(245,248,245,.7);
        font-size: .76rem;
        border-top: 1px solid rgba(31,138,76,.35);
        padding-top: .75rem;
        margin-top: .65rem;
    }

    .leaf-art {
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border: 1px dashed rgba(31,138,76,.65);
        border-radius: 16px;
        background: rgba(31,138,76,.055);
        padding: 1.2rem;
    }

    .leaf-art .leaf-symbol {
        color: var(--green);
        font-size: 4.2rem;
        line-height: 1;
    }

    .leaf-art strong {
        color: var(--white);
        font-size: 1.05rem;
        margin-top: .7rem;
    }

    .leaf-art p {
        color: rgba(245,248,245,.62);
        font-size: .8rem;
        max-width: 270px;
    }

    /* UPLOAD */

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(31,138,76,.075) !important;
        border: 1px dashed var(--green) !important;
        border-radius: 15px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: var(--white) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: var(--green) !important;
        color: var(--white) !important;
        border: 1px solid var(--green) !important;
        border-radius: 9px !important;
    }

    [data-testid="stCheckbox"] label {
        color: var(--white) !important;
    }

    [data-testid="stCheckbox"] input:checked + div {
        background: var(--green) !important;
    }

    /* BUTTONS */

    [data-testid="stButton"] button {
        background: var(--green) !important;
        color: var(--white) !important;
        border: 1px solid var(--green) !important;
        border-radius: 10px !important;
        font-weight: 850 !important;
        letter-spacing: .04em !important;
        transition: transform .15s ease, filter .15s ease;
    }

    [data-testid="stButton"] button:hover {
        filter: brightness(1.12);
        transform: translateY(-1px);
    }

    /* METRICS / PROGRESS */

    [data-testid="stProgressBar"] > div > div {
        background: var(--green) !important;
    }

    [data-testid="stProgressBar"] > div {
        background: rgba(245,248,245,.13) !important;
    }

    [data-testid="stMetric"] {
        background: var(--black-soft);
        border: 1px solid rgba(31,138,76,.48);
        padding: .9rem 1rem;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: rgba(245,248,245,.7) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--white) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(31,138,76,.4) !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary {
        color: var(--white) !important;
    }

    /* ALERTS */

    .alert {
        border-radius: 13px;
        padding: .9rem 1.05rem;
        margin: .55rem 0;
        border: 1px solid;
        line-height: 1.5;
        font-size: .86rem;
    }

    .alert-red {
        background: rgba(217,45,62,.10);
        border-color: var(--red);
        color: var(--white);
    }

    .alert-green {
        background: rgba(31,138,76,.12);
        border-color: var(--green);
        color: var(--white);
    }

    .alert-dark {
        background: var(--black-soft);
        border-color: rgba(245,248,245,.22);
        color: var(--white);
    }

    .alert b {
        letter-spacing: .04em;
    }

    /* RESULT CARDS */

    .result-card {
        background:
            linear-gradient(
                125deg,
                rgba(31,138,76,.23),
                rgba(7,10,7,.12) 75%
            ),
            var(--black-soft);
        border: 1px solid var(--green);
        border-radius: 19px;
        padding: 1.2rem;
        min-height: 170px;
    }

    .result-label {
        color: var(--green);
        font-size: .68rem;
        font-weight: 950;
        letter-spacing: .16em;
    }

    .result-name {
        color: var(--white);
        font-size: clamp(1.35rem, 2.4vw, 2.15rem);
        font-weight: 900;
        line-height: 1.08;
        margin: .55rem 0;
    }

    .result-score {
        color: var(--white);
        font-size: 2.1rem;
        font-weight: 950;
        letter-spacing: -.05em;
    }

    .rank-row {
        margin: .9rem 0;
    }

    .rank-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        color: var(--white);
        font-size: .83rem;
        margin-bottom: .34rem;
    }

    .rank-track {
        height: 9px;
        border-radius: 99px;
        background: rgba(245,248,245,.13);
        overflow: hidden;
    }

    .rank-fill {
        height: 100%;
        border-radius: 99px;
        background: var(--green);
    }

    .section-divider {
        height: 1px;
        background: rgba(31,138,76,.4);
        margin: 1.25rem 0;
    }

    .footer-line {
        border-top: 1px solid rgba(31,138,76,.4);
        margin-top: 2rem;
        padding-top: .85rem;
        color: rgba(245,248,245,.55);
        font-size: .72rem;
        letter-spacing: .05em;
    }

    @media (max-width: 760px) {
        .step-strip {
            grid-template-columns: repeat(2, 1fr);
        }

        .brand-row {
            align-items: flex-start;
            flex-direction: column;
        }

        .hero-title {
            font-size: 3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------- Helper functions ----------------------------

def render_alert(
    message: str,
    level: str = "dark",
    heading: str | None = None,
) -> None:
    """Render a branded alert using the requested palette."""
    safe_message = html.escape(message)

    safe_heading = (
        f"<b>{html.escape(heading)} · </b>"
        if heading else ""
    )

    css_class = {
        "red": "alert-red",
        "green": "alert-green",
        "dark": "alert-dark",
    }.get(level, "alert-dark")

    st.markdown(
        f'<div class="alert {css_class}">'
        f'{safe_heading}{safe_message}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model(model_path: str, modified_time: float):
    del modified_time

    loaded = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if not loaded.inputs or not loaded.outputs:
        raise ValueError(
            "The saved model does not expose valid inputs and outputs."
        )

    return loaded


@st.cache_data(show_spinner=False)
def load_mapping(
    mapping_path: str,
    modified_time: float,
) -> dict[str, Any]:

    del modified_time

    with open(mapping_path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "The class mapping JSON must contain a non-empty object."
        )

    return mapping


def class_names_from_mapping(
    mapping: dict[str, Any],
    count: int,
) -> list[str]:

    if all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in mapping.values()
    ):

        names: list[str | None] = [None] * count

        for class_name, class_id in mapping.items():
            idx = int(class_id)

            if 0 <= idx < count:
                names[idx] = str(class_name)

        if any(name is None for name in names):
            missing = [
                str(i)
                for i, name in enumerate(names)
                if name is None
            ]

            raise ValueError(
                "Class mapping is missing model output IDs: "
                + ", ".join(missing[:10])
            )

        return [str(name) for name in names]

    if all(str(key).isdigit() for key in mapping):
        names = [
            str(mapping.get(str(i), ""))
            for i in range(count)
        ]

        if any(not name for name in names):
            raise ValueError(
                "Class mapping does not contain every model output ID."
            )

        return names

    raise ValueError("Unrecognized class_mapping.json format.")


def friendly_label(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ")


def read_uploaded_image(
    raw: bytes,
    filename: str = "upload",
) -> tuple[Image.Image, list[str]]:

    if not raw:
        raise ValueError("The uploaded file is empty.")

    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(
            f"File is larger than the {MAX_UPLOAD_MB} MB upload limit."
        )

    warnings: list[str] = []

    try:
        with Image.open(io.BytesIO(raw)) as check:

            image_format = (check.format or "unknown").upper()

            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise ValueError(
                    f"Detected image format '{image_format}' is not supported. "
                    "Use JPG, PNG, BMP, WEBP, or TIFF."
                )

            if getattr(check, "n_frames", 1) > 1:
                warnings.append(
                    "This file contains multiple frames; "
                    "only the first frame is used."
                )

            check.verify()

        suffix = Path(filename).suffix.lower()
        expected_formats = EXTENSION_TO_FORMATS.get(suffix)

        if expected_formats and image_format not in expected_formats:
            warnings.append(
                f"The filename ends in '{suffix}', but the file contents "
                f"are {image_format}. The app checks actual contents "
                "rather than trusting the filename."
            )

        with Image.open(io.BytesIO(raw)) as image:

            image = ImageOps.exif_transpose(image)
            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError("Image dimensions are invalid.")

            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions are unusually large "
                    f"({width:,} × {height:,}). Please resize it."
                )

            if min(width, height) < MIN_IMAGE_SIDE:
                raise ValueError(
                    f"Image is too small ({width} × {height}). "
                    f"Upload an image at least {MIN_IMAGE_SIDE} "
                    "pixels wide and tall."
                )

            if image.mode in ("L", "1"):
                warnings.append(
                    "This is a grayscale image. The model was trained "
                    "on color images, so results may be less reliable."
                )

            elif "A" in image.getbands():
                warnings.append(
                    "Transparency was removed by converting the image to RGB."
                )

            image = image.convert("RGB")
            image.load()

            return image.copy(), warnings

    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as exc:

        raise ValueError(
            "This file is not a readable image, or the image is damaged. "
            "Renaming a corrupted file to .jpg does not make it valid. "
            "Open the original and export it again as JPG or PNG."
        ) from exc


def basic_quality_warnings(
    image: Image.Image,
) -> list[str]:

    arr = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    warnings: list[str] = []

    if float(arr.std()) < 8.0:
        warnings.append(
            "The image has very little visual contrast; "
            "it may be blank, overexposed, or heavily blurred."
        )

    gray = arr.mean(axis=2)

    if gray.shape[0] > 2 and gray.shape[1] > 2:

        edge_energy = float(
            np.abs(np.diff(gray, axis=0)).mean()
            + np.abs(np.diff(gray, axis=1)).mean()
        )

        if edge_energy < 2.0:
            warnings.append(
                "The image looks unusually smooth or blurry. "
                "A sharper, well-lit leaf photo may work better."
            )

    return warnings


def make_input_batch(
    image: Image.Image,
) -> tuple[Image.Image, np.ndarray]:

    resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    )

    processed = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            array.copy()
        )
    )

    return resized, np.expand_dims(processed, axis=0)


def render_rank_bar(
    rank: int,
    label: str,
    value: float,
) -> None:

    pct = float(np.clip(value, 0.0, 1.0)) * 100.0

    st.markdown(
        f"""
        <div class="rank-row">
          <div class="rank-head">
            <span><b>{rank:02d}</b> &nbsp;
            {html.escape(label)}</span>
            <b>{value * 100:.2f}%</b>
          </div>
          <div class="rank-track">
            <div class="rank-fill" style="width:{pct:.2f}%"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------- Sidebar ----------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="padding:.5rem 0 1rem 0">
          <div class="brand-mark">
            <span class="brand-icon">L</span>
            LEAF / LENS
          </div>
          <div class="eyebrow" style="margin-top:.85rem">
            AI crop screening
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### SYSTEM PROFILE")

    st.markdown(
        """
        <div class="hero-stat">
            <b>MobileNetV2</b>
            <small>MODEL ARCHITECTURE</small>
        </div>
        <div class="hero-stat">
            <b>COLOR</b>
            <small>TRAINING IMAGE SECTION</small>
        </div>
        <div class="hero-stat">
            <b>224 × 224</b>
            <small>MODEL INPUT SIZE</small>
        </div>
        <div class="hero-stat">
            <b>38 LABELS</b>
            <small>PLANTVILLAGE CLASSES</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-divider'></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### SESSION GUIDE")

    st.markdown(
        """
        <div class="muted">01 &nbsp; Upload a clear leaf photo</div>
        <div class="muted">02 &nbsp; Confirm supported input</div>
        <div class="muted">03 &nbsp; Review model candidates</div>
        <div class="muted">04 &nbsp; Inspect Grad-CAM</div>
        """,
        unsafe_allow_html=True,
    )

    render_alert(
        "The app does not automatically identify non-leaf photos. "
        "A model score is not a calibrated probability or a confirmed diagnosis.",
        level="red",
        heading="RESEARCH PROTOTYPE",
    )


# -------------------------------- Main header ------------------------------

head_left, head_right = st.columns(
    [1.55, 0.7],
    gap="large",
)

with head_left:

    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">
            <span class="brand-icon">L</span>
            CROP DISEASE INTELLIGENCE
          </div>
          <div class="status-pill">
            <span class="status-dot"></span>
            SCREENING PROTOTYPE
          </div>
        </div>

        <div class="hero">
          <div class="eyebrow">
            COMPUTER VISION / PLANT HEALTH
          </div>
          <div class="hero-title">
            LeafLine🧬<br>
            <span>IN FOCUS.</span>
          </div>
          <div class="hero-copy">
            Upload a clear plant-leaf image to explore the
            model's predicted PlantVillage class, compare
            its top candidates, and inspect a visual explanation.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with head_right:

    st.markdown(
        """
        <div class="hero-side" style="margin-top:3.7rem">
          <div class="eyebrow">MODEL SNAPSHOT</div>

          <div class="hero-stat">
            <b>38</b>
            <small>SUPPORTED TRAINING LABELS</small>
          </div>

          <div class="hero-stat">
            <b>3 CHANNELS</b>
            <small>RGB COLOR INPUT</small>
          </div>

          <div class="hero-stat">
            <b>224²</b>
            <small>STANDARD IMAGE WINDOW</small>
          </div>

          <div class="hero-stat">
            <b>GRAD-CAM</b>
            <small>OPTIONAL VISUAL EXPLANATION</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="step-strip">
      <div class="step-item">
        <span class="step-no">01</span> INPUT IMAGE
      </div>
      <div class="step-item">
        <span class="step-no">02</span> VALIDATE
      </div>
      <div class="step-item">
        <span class="step-no">03</span> CLASSIFY
      </div>
      <div class="step-item">
        <span class="step-no">04</span> EXPLAIN
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Reset before creating the uploader widget.

reset_col, reset_note_col = st.columns([0.8, 2.2])

with reset_col:

    if st.button("RESET / NEW SCAN", use_container_width=True):

        for key in (
            "leaf_image_uploader",
            "confirm_plant_leaf",
            "_active_upload_signature",
        ):
            st.session_state.pop(key, None)

        st.rerun()

with reset_note_col:

    st.markdown(
        "<p class='muted' style='padding:.55rem 0'>"
        "Start a fresh scan at any time. Your model files remain unchanged."
        "</p>",
        unsafe_allow_html=True,
    )


# -------------------------------- Upload UI --------------------------------

upload_col, preview_col = st.columns(
    [1.05, 0.95],
    gap="large",
)

with upload_col:

    st.markdown(
        '<div class="panel-title">01 / UPLOAD YOUR LEAF IMAGE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">Choose a close, sharp, well-lit leaf photograph.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop image here or browse files",
        type=[
            "jpg", "jpeg", "png",
            "bmp", "webp", "tif",
            "tiff", "gif",
        ],
        key="leaf_image_uploader",
        help=(
            f"Maximum file size: {MAX_UPLOAD_MB} MB. "
            "Supported: JPG, PNG, BMP, WEBP, TIFF, GIF."
        ),
    )

    st.markdown(
        f'<div class="upload-note">'
        f'MAX SIZE: {MAX_UPLOAD_MB} MB &nbsp; · &nbsp; '
        f'MINIMUM SIDE: {MIN_IMAGE_SIDE}px &nbsp; · &nbsp; '
        'RGB MODEL INPUT</div>',
        unsafe_allow_html=True,
    )


image: Image.Image | None = None
warnings: list[str] = []
uploaded_bytes = b""

if uploaded_file is not None:

    try:

        uploaded_bytes = uploaded_file.getvalue()

        image, warnings = read_uploaded_image(
            uploaded_bytes,
            uploaded_file.name,
        )

        warnings.extend(
            basic_quality_warnings(image)
        )

        upload_signature = hashlib.sha256(
            uploaded_bytes
        ).hexdigest()

        if (
            st.session_state.get("_active_upload_signature")
            != upload_signature
        ):

            st.session_state["confirm_plant_leaf"] = False

            st.session_state[
                "_active_upload_signature"
            ] = upload_signature

    except ValueError as exc:

        with preview_col:

            render_alert(
                str(exc),
                level="red",
                heading="UPLOAD REJECTED",
            )

            render_alert(
                "For a damaged file, open the original in an image viewer "
                "and export a new JPG or PNG. Changing the extension does "
                "not repair corrupted image data.",
                level="dark",
                heading="HOW TO FIX",
            )

        st.stop()

    except Exception as exc:

        LOGGER.exception(
            "Unexpected image decoding failure"
        )

        with preview_col:

            render_alert(
                "The uploaded file could not be processed. "
                "Please try another image.",
                level="red",
                heading="IMAGE ERROR",
            )

            with st.expander("Technical details"):
                st.code(
                    f"{type(exc).__name__}: {exc}"
                )

        st.stop()


# -------------------------------- Preview UI -------------------------------

with preview_col:

    st.markdown(
        '<div class="panel-title">02 / IMAGE PREVIEW</div>',
        unsafe_allow_html=True,
    )

    if image is None:

        st.markdown(
            """
            <div class="leaf-art">
              <div class="leaf-symbol">✳</div>
              <strong>READY FOR YOUR SCAN</strong>
              <p>
                Your uploaded leaf image will appear here
                with its dimensions and file details.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.image(
            image,
            caption="Uploaded image · normalized to RGB",
            use_container_width=True,
        )

        st.markdown(
            f"""
            <p class="muted">
              {html.escape(uploaded_file.name)}<br>
              {image.width:,} × {image.height:,} px &nbsp; · &nbsp;
              {len(uploaded_bytes) / (1024 * 1024):.2f} MB
            </p>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------- Empty state -------------------------------

if uploaded_file is None or image is None:

    st.markdown(
        "<div class='section-divider'></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### BEFORE YOU SCAN")

    guide_cols = st.columns(3)

    guide_items = [
        (
            "01",
            "Frame one leaf",
            "Keep the leaf clearly visible and close to the camera.",
        ),
        (
            "02",
            "Use natural light",
            "Avoid strong shadows, glare and heavy blur.",
        ),
        (
            "03",
            "Check the result",
            "Treat the output as a screening aid, not a final diagnosis.",
        ),
    ]

    for column, (number, title, description) in zip(
        guide_cols,
        guide_items,
    ):

        with column:

            st.markdown(
                f"""
                <div class="panel">
                    <div class="eyebrow">{number}</div>
                    <div class="panel-title">{title}</div>
                    <p class="muted">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="footer-line">'
        'LEAF / LENS &nbsp; · &nbsp; MOBILENETV2 &nbsp; · &nbsp; '
        'PLANTVILLAGE COLOR MODEL'
        '</div>',
        unsafe_allow_html=True,
    )

    st.stop()


# --------------------------- Upload quality notes -------------------------

for warning in warnings:

    render_alert(
        warning,
        level="red",
        heading="IMAGE QUALITY NOTE",
    )


# ---------------------------- Leaf confirmation ---------------------------

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-kicker">03 / INPUT CONFIRMATION</div>',
    unsafe_allow_html=True,
)

leaf_confirmed = st.checkbox(
    "I confirm this upload shows a plant leaf "
    "(not a person, animal, object, screenshot, or scenery).",
    key="confirm_plant_leaf",
    help=(
        "The app cannot reliably identify non-leaf photos yet. "
        "Do not continue if you are unsure."
    ),
)

if not leaf_confirmed:

    render_alert(
        "Prediction is paused. Confirm that this image contains a "
        "plant leaf before continuing. This checkbox is not automatic "
        "image recognition.",
        level="red",
        heading="LEAF CONFIRMATION REQUIRED",
    )

    st.stop()


# ------------------------------- Model checks ------------------------------

if not MODEL_PATH.is_file():

    render_alert(
        "The trained model file was not found.",
        level="red",
        heading="MODEL FILE MISSING",
    )

    st.code(str(MODEL_PATH))
    st.stop()


if not CLASS_MAPPING_PATH.is_file():

    render_alert(
        "The class mapping file was not found.",
        level="red",
        heading="CLASS MAPPING MISSING",
    )

    st.code(str(CLASS_MAPPING_PATH))
    st.stop()


if GRADCAM_IMPORT_ERROR is not None:

    render_alert(
        "Grad-CAM helper could not be imported. Prediction may still work.",
        level="red",
        heading="VISUAL EXPLANATION UNAVAILABLE",
    )

    with st.expander("Grad-CAM import details"):

        st.code(
            f"{type(GRADCAM_IMPORT_ERROR).__name__}: "
            f"{GRADCAM_IMPORT_ERROR}"
        )


# -------------------------------- Inference ---------------------------------

try:

    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    mapping = load_mapping(
        str(CLASS_MAPPING_PATH),
        CLASS_MAPPING_PATH.stat().st_mtime,
    )

    resized_image, image_batch = make_input_batch(
        image
    )

    with st.spinner("Analyzing leaf image..."):

        prediction_result = model.predict(
            image_batch,
            verbose=0,
        )

    prediction_result = np.asarray(
        prediction_result
    )

    if (
        prediction_result.ndim != 2
        or prediction_result.shape[0] != 1
    ):

        raise ValueError(
            f"Unexpected model output shape: "
            f"{prediction_result.shape}; expected "
            "(1, number_of_classes)."
        )

    prediction_vector = prediction_result[0].astype(
        np.float64
    )

    if (
        prediction_vector.size == 0
        or not np.all(np.isfinite(prediction_vector))
    ):

        raise ValueError(
            "The model returned empty or non-finite prediction scores."
        )

    if (
        np.any(prediction_vector < -1e-5)
        or np.any(prediction_vector > 1.00001)
    ):

        render_alert(
            "Model outputs are outside the usual 0–1 range. "
            "Treat the displayed values as raw model outputs.",
            level="red",
            heading="OUTPUT RANGE NOTE",
        )

    class_names = class_names_from_mapping(
        mapping,
        len(prediction_vector),
    )

    if len(class_names) != len(prediction_vector):

        raise ValueError(
            f"Mapping has {len(class_names)} classes but "
            f"the model returned {len(prediction_vector)} scores."
        )

    predicted_index = int(
        np.argmax(prediction_vector)
    )

    predicted_class = class_names[
        predicted_index
    ]

    score = float(
        prediction_vector[predicted_index]
    )

except Exception as exc:

    LOGGER.exception("Inference failed")

    render_alert(
        "The model could not analyze this image. "
        "The app is still running; try another image or inspect "
        "the technical details.",
        level="red",
        heading="ANALYSIS FAILED",
    )

    with st.expander(
        "Technical details for troubleshooting",
        expanded=True,
    ):

        st.code(
            f"{type(exc).__name__}: {exc}"
        )

    st.stop()


# ------------------------------- Grad-CAM ----------------------------------

overlay_image: Image.Image | None = None
heatmap_image: Image.Image | None = None
gradcam_error: Exception | None = None

if (
    make_gradcam is not None
    and save_visualizations is not None
):

    try:

        with st.spinner("Generating visual explanation..."):

            heatmap = make_gradcam(
                model,
                image_batch,
                predicted_index,
            )

            with tempfile.TemporaryDirectory(
                prefix="crop_gradcam_"
            ) as temp_dir:

                output_dir = Path(temp_dir)

                save_visualizations(
                    resized_image,
                    heatmap,
                    output_dir,
                )

                overlay_path = (
                    output_dir / "gradcam_overlay.png"
                )

                heatmap_path = (
                    output_dir / "gradcam_heatmap.png"
                )

                if (
                    not overlay_path.is_file()
                    or not heatmap_path.is_file()
                ):

                    raise FileNotFoundError(
                        "Grad-CAM helper did not create "
                        "the expected image files."
                    )

                overlay_image = (
                    Image.open(overlay_path)
                    .convert("RGB")
                    .copy()
                )

                heatmap_image = (
                    Image.open(heatmap_path)
                    .convert("RGB")
                    .copy()
                )

    except Exception as exc:

        LOGGER.exception(
            "Grad-CAM generation failed"
        )

        gradcam_error = exc


# ------------------------------- Results UI --------------------------------

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-kicker">04 / ANALYSIS REPORT</div>',
    unsafe_allow_html=True,
)

st.markdown("## Your screening result")

result_col, score_col, class_col = st.columns(
    [1.45, 0.8, 0.8],
    gap="medium",
)

with result_col:

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">TOP PREDICTED CLASS</div>
          <div class="result-name">
            {html.escape(friendly_label(predicted_class))}
          </div>
          <div class="muted">
            Highest-scoring class from the supported PlantVillage labels
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with score_col:

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">MODEL SCORE</div>
          <div class="result-score">{score * 100:.2f}%</div>
          <div class="muted">Raw model output; not calibrated</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with class_col:

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">CLASS SPACE</div>
          <div class="result-score">{len(class_names)}</div>
          <div class="muted">Supported PlantVillage classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if score < LOW_SCORE_WARNING:

    render_alert(
        "The top model score is relatively low. Treat this result "
        "as uncertain and try a clearer leaf image or seek expert advice.",
        level="red",
        heading="LOW-SCORE WARNING",
    )


render_alert(
    "The upload passed file checks and your confirmation, but the "
    "app has not independently verified that it is a leaf. If the "
    "photo is not a leaf, this prediction is invalid. Automatic "
    "rejection requires a separate leaf/non-leaf detector.",
    level="red",
    heading="IMPORTANT LIMITATION",
)

st.caption(
    "A high score is not proof that the diagnosis is correct. "
    "Use the result as a preliminary screening signal only."
)


# ----------------------------- Top predictions -----------------------------

left_results, right_results = st.columns(
    [1.05, 0.95],
    gap="large",
)

with left_results:

    st.markdown("### TOP 5 CANDIDATES")

    st.markdown(
        '<p class="muted">Ranked by the model output for this upload.</p>',
        unsafe_allow_html=True,
    )

    top_indices = np.argsort(
        prediction_vector
    )[::-1][:min(5, len(prediction_vector))]

    for rank, idx in enumerate(
        top_indices,
        start=1,
    ):

        render_rank_bar(
            rank,
            friendly_label(
                class_names[int(idx)]
            ),
            float(prediction_vector[int(idx)]),
        )


# --------------------------- Grad-CAM presentation -------------------------

with right_results:

    st.markdown("### VISUAL EXPLANATION")

    if overlay_image is not None:

        st.image(
            overlay_image,
            caption="Grad-CAM overlay",
            use_container_width=True,
        )

        if heatmap_image is not None:

            with st.expander("VIEW RAW HEATMAP"):

                st.image(
                    heatmap_image,
                    caption="Raw Grad-CAM heatmap",
                    use_container_width=True,
                )

        render_alert(
            "Grad-CAM highlights image regions associated with this "
            "prediction. It does not prove the model used valid "
            "disease features.",
            level="dark",
            heading="HOW TO READ THIS",
        )

    elif gradcam_error is not None:

        render_alert(
            "The prediction succeeded, but Grad-CAM could not be "
            "generated for this image/model.",
            level="red",
            heading="GRAD-CAM UNAVAILABLE",
        )

        with st.expander("Grad-CAM troubleshooting details"):

            st.code(
                f"{type(gradcam_error).__name__}: "
                f"{gradcam_error}"
            )

    else:

        render_alert(
            "The Grad-CAM helper module did not load. "
            "The prediction above is still available.",
            level="red",
            heading="GRAD-CAM UNAVAILABLE",
        )


# ------------------------------ Interpretation ----------------------------

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True,
)

st.markdown("### HOW TO INTERPRET THIS RESULT")

interpret_cols = st.columns(3)

interpretations = [
    (
        "MODEL CANDIDATE",
        "The top label is the model's highest-scoring class, "
        "not a confirmed diagnosis.",
    ),
    (
        "SCORE",
        "The displayed score is the model output and may not "
        "be a calibrated probability.",
    ),
    (
        "NEXT STEP",
        "Compare with visible symptoms and consult a qualified "
        "agricultural expert before acting.",
    ),
]

for column, (title, body) in zip(
    interpret_cols,
    interpretations,
):

    with column:

        st.markdown(
            f"""
            <div class="panel">
                <div class="eyebrow">
                    {html.escape(title)}
                </div>
                <p class="muted">
                    {html.escape(body)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------- Footer ----------------------------------

st.markdown(
    '<div class="footer-line">'
    'LEAF / LENS &nbsp; · &nbsp; MOBILENETV2 &nbsp; · &nbsp; '
    'PLANTVILLAGE COLOR MODEL &nbsp; · &nbsp; 224 × 224 INPUT'
    '</div>',
    unsafe_allow_html=True,
)
