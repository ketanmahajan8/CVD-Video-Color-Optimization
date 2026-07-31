import sys
import os
import tempfile
import subprocess
from pathlib import Path
import csv

import cv2
import imageio_ffmpeg as ffmpeg
import streamlit as st
from PIL import Image

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from inference.video_pipeline import process_video


# -----------------------------
# Paths (LOCAL)
# -----------------------------
CKPT_PATH = ROOT_DIR / "checkpoints" / "enhance_gan_conditional" / "latest.pth"
PLATE_DIR = ROOT_DIR / "app" / "plates"
OUTPUT_DIR = ROOT_DIR / "outputs" / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_LIST = ["protan", "deutan", "tritan"]
PLATE_CONFIG_PATH = PLATE_DIR / "plates_config.csv"

# -----------------------------
# Session State
# -----------------------------
if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

if "original_frames" not in st.session_state:
    st.session_state.original_frames = []

if "enhanced_frames" not in st.session_state:
    st.session_state.enhanced_frames = []

if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

if "enhanced_bytes" not in st.session_state:
    st.session_state.enhanced_bytes = None

if "download_bytes" not in st.session_state:
    st.session_state.download_bytes = None

if "out_path" not in st.session_state:
    st.session_state.out_path = None

if "preview_path" not in st.session_state:
    st.session_state.preview_path = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None


# -----------------------------
# Synthetic Plates Definition
# -----------------------------
def load_plates_config():
    if not PLATE_CONFIG_PATH.exists():
        st.error(f"Missing plate configuration file: {PLATE_CONFIG_PATH}")
        st.stop()

    plates = []

    with open(PLATE_CONFIG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            file_name = row.get("file", "").strip()
            correct = row.get("correct", "").strip()
            axis = row.get("axis", "").strip().lower()

            if not file_name or not correct or axis not in ["rg", "by"]:
                st.error(
                    "Invalid row in plates_config.csv. "
                    "Each row must have: file, correct, axis where axis is rg or by."
                )
                st.stop()

            plates.append({
                "file": file_name,
                "correct": correct,
                "axis": axis,
            })

    return plates


PLATES = load_plates_config()


def classify_from_scores(rg_fail: int, by_fail: int):
    """
    Screening-based classification.
    This is only used for selecting the enhancement mode.
    """
    rg_thresh = 2
    by_thresh = 2

    if by_fail >= by_thresh and by_fail > rg_fail:
        return "tritan"

    if rg_fail >= rg_thresh and rg_fail > by_fail:
        return "red-green"

    if rg_fail < rg_thresh and by_fail < by_thresh:
        return "normal"

    return "uncertain"


def extract_frames(video_path: str, frame_count: int = 3):
    """
    Extracts representative frames from a video:
    beginning, middle, and near end.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    if frame_count == 3:
        positions = [0, total // 2, max(total - 1, 0)]
    else:
        positions = [
            int(i * (total - 1) / max(frame_count - 1, 1))
            for i in range(frame_count)
        ]

    frames = []

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def convert_to_browser_mp4(input_path: str, output_path: str) -> bool:
    """
    Converts OpenCV MP4 output into browser-friendly H.264 MP4.
    This is only for Streamlit preview compatibility.
    The original processed video is still kept for download fallback.
    """
    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return Path(output_path).exists() and Path(output_path).stat().st_size > 0

    except Exception:
        return False


def show_architecture_diagram():
    st.subheader("System Architecture")

    c1, a1, c2, a2, c3, a3, c4, a4, c5 = st.columns(
        [2, 0.4, 2, 0.4, 2, 0.4, 2, 0.4, 2]
    )

    with c1:
        st.info("Screening Test\n\nSynthetic Ishihara Plates")

    with a1:
        st.markdown("### →")

    with c2:
        st.info("Type Selection\n\nProtan / Deutan / Tritan")

    with a2:
        st.markdown("### →")

    with c3:
        st.info("Video Upload\n\nInput MP4")

    with a3:
        st.markdown("### →")

    with c4:
        st.info("Conditional GAN\n\nFrame Enhancement")

    with a4:
        st.markdown("### →")

    with c5:
        st.info("Result Module\n\nComparison / Download")


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="CVD Video Pre-Compensation", layout="centered")

st.title("CVD Video Pre-Compensation Using GAN")

st.info(
    "This application uses a lightweight color-vision screening test to suggest an enhancement mode. "
    "The result is used only for model selection and is not a medical diagnosis."
)

show_architecture_diagram()

with st.expander("Project Workflow", expanded=False):
    st.markdown(
        """
        1. **Screening Test:** The user answers synthetic Ishihara-style plates.
        2. **Deficiency Estimation:** The app estimates whether the difficulty is on the red-green or blue-yellow axis.
        3. **Enhancement Type Selection:** The user can accept the suggested type or manually select Protan, Deutan, or Tritan.
        4. **Video Processing:** Each video frame is passed through the trained conditional GAN.
        5. **Output Generation:** The enhanced frames are written back into a processed video.
        6. **Result Comparison:** The app shows original vs enhanced video preview and representative frame comparisons.
        7. **Download:** The final enhanced video can be downloaded for local playback or demonstration.
        """
    )

st.subheader("Screening Plate Control")

if st.button("Generate New Screening Test"):
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "app" / "generate_synthetic_plates.py")],
        check=True,
    )

    st.session_state.result_ready = False
    st.success("New screening plates generated successfully.")
    st.rerun()

# ---------- Screening ----------
st.header("Step 1: Screening Test")
st.write("Enter the number you see for each plate. If the number is unclear, type 'none'.")

rg_fail = 0
by_fail = 0
answered = 0

with st.expander("Take Screening Test", expanded=True):
    for idx, plate in enumerate(PLATES):
        img_path = PLATE_DIR / plate["file"]

        if not img_path.exists():
            st.error(f"Missing plate image: {img_path}")
            st.stop()

        st.image(
            Image.open(img_path),
            caption=f"Plate {idx + 1}",
            use_container_width=True,
        )

        ans = st.text_input(f"Answer for Plate {idx + 1}", key=f"plate_{idx}").strip()

        if ans != "":
            answered += 1
            ans_norm = ans.lower().replace(" ", "")
            correct_norm = plate["correct"].lower()

            ok = ans_norm == correct_norm

            if not ok:
                if plate["axis"] == "rg":
                    rg_fail += 1
                else:
                    by_fail += 1

st.write(f"Red-Green axis misses: **{rg_fail} / 6**")
st.write(f"Blue-Yellow axis misses: **{by_fail} / 6**")

suggestion = classify_from_scores(rg_fail, by_fail)

if st.button("Get Suggested Enhancement Type"):
    st.session_state["suggestion"] = suggestion

    if suggestion == "tritan":
        st.warning("Suggested enhancement type: **Tritan**")
    elif suggestion == "red-green":
        st.warning("Suggested enhancement type: **Red-Green deficiency**. Defaulting to Deutan mode.")
    elif suggestion == "normal":
        st.success("No strong deficiency pattern detected by the screening test.")
    else:
        st.warning("Screening result is uncertain. Please select the enhancement type manually.")


# ---------- Choose type ----------
st.header("Step 2: Select Enhancement Type and Upload Video")

suggestion = st.session_state.get("suggestion", "uncertain")

if suggestion == "tritan":
    default_type = "tritan"
elif suggestion == "red-green":
    default_type = "deutan"
else:
    default_type = "protan"

cvd_type = st.selectbox(
    "Enhancement type",
    TYPE_LIST,
    index=TYPE_LIST.index(default_type),
)

alpha = st.slider(
    "Color Enhancement Level",
    0.5,
    1.0,
    0.9,
    0.05,
    help="Controls how strongly the enhanced GAN output is blended with the original video.",
)

smooth = st.slider(
    "Video Stability Control",
    0.0,
    0.6,
    0.35,
    0.05,
    help="Reduces frame-to-frame flickering by smoothing consecutive enhanced frames.",
)

st.subheader("Output Quality Controls")
st.caption("Adjust output clarity and enhancement parameters to improve visual quality.")

sharpen = st.slider(
    "Image Clarity",
    0.0,
    1.0,
    0.5,
    0.05,
    help="Controls the amount of edge enhancement applied after GAN processing.",
)

sharpen_radius = st.slider(
    "Clarity Detail Size",
    0.5,
    2.5,
    1.0,
    0.1,
    help="Controls the size of details affected by the clarity adjustment.",
)

sharpen_thresh = st.slider(
    "Noise Protection Threshold",
    0,
    10,
    3,
    1,
    help="Prevents sharpening from affecting flat or noisy regions too strongly.",
)

uploaded = st.file_uploader(
    "Upload MP4 video",
    type=["mp4"],
)

if uploaded is not None:
    if st.session_state.last_uploaded_name != uploaded.name:
        st.session_state.result_ready = False
        st.session_state.original_frames = []
        st.session_state.enhanced_frames = []
        st.session_state.uploaded_bytes = None
        st.session_state.enhanced_bytes = None
        st.session_state.download_bytes = None
        st.session_state.out_path = None
        st.session_state.preview_path = None
        st.session_state.last_uploaded_name = uploaded.name


# ---------- Run ----------
if uploaded and st.button("Process Video"):
    if not Path(CKPT_PATH).exists():
        st.error(f"Missing checkpoint: {CKPT_PATH}")
        st.stop()

    uploaded_bytes = uploaded.read()

    st.info("Processing video. Please wait...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_bytes)
        in_path = tmp_in.name

    out_path = str(OUTPUT_DIR / f"streamlit_out_{cvd_type}.mp4")
    preview_path = str(OUTPUT_DIR / f"streamlit_preview_{cvd_type}.mp4")

    def on_progress(done, total):
        if total is None:
            status_text.text(f"Processed {done} frames...")
            progress_bar.progress(min(99, done % 100))
        else:
            pct = int((done / total) * 100)
            progress_bar.progress(min(100, pct))
            status_text.text(f"Processing {done}/{total} frames ({pct}%)")

    try:
        process_video(
            in_video=in_path,
            out_video=out_path,
            ckpt_path=CKPT_PATH,
            cvd_type=cvd_type,
            model_size=256,
            alpha=float(alpha),
            smooth=float(smooth),
            lab_boost=0.25,
            sharpen=float(sharpen),
            sharpen_radius=float(sharpen_radius),
            sharpen_thresh=int(sharpen_thresh),
            progress_callback=on_progress,
        )

        progress_bar.progress(100)
        status_text.text("Processing complete (100%)")

        original_frames = extract_frames(in_path, frame_count=3)
        enhanced_frames = extract_frames(out_path, frame_count=3)

        with open(out_path, "rb") as f:
            download_bytes = f.read()

        preview_ok = convert_to_browser_mp4(out_path, preview_path)

        if preview_ok:
            with open(preview_path, "rb") as f:
                enhanced_bytes = f.read()
        else:
            enhanced_bytes = download_bytes
            st.warning(
                "Browser preview conversion could not be completed. "
                "Frame comparison and download are still available."
            )

        st.session_state.result_ready = True
        st.session_state.original_frames = original_frames
        st.session_state.enhanced_frames = enhanced_frames
        st.session_state.uploaded_bytes = uploaded_bytes
        st.session_state.enhanced_bytes = enhanced_bytes
        st.session_state.download_bytes = download_bytes
        st.session_state.out_path = out_path
        st.session_state.preview_path = preview_path if preview_ok else None
        st.session_state.last_uploaded_name = uploaded.name

    finally:
        try:
            os.remove(in_path)
        except:
            pass

    st.success("Video processing completed successfully.")


# ---------- Persistent Results Section ----------
if st.session_state.result_ready:
    st.header("Step 3: Result Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Video")
        st.video(st.session_state.uploaded_bytes)

    with col2:
        st.subheader("Enhanced Video")
        st.video(st.session_state.enhanced_bytes)

    if st.session_state.original_frames and st.session_state.enhanced_frames:
        st.header("Frame-Level Comparison")
        st.caption("Representative frames are shown for visual comparison.")

        labels = ["Beginning Frame", "Middle Frame", "Ending Frame"]

        total_pairs = min(
            len(st.session_state.original_frames),
            len(st.session_state.enhanced_frames),
        )

        for i in range(total_pairs):
            st.subheader(labels[i] if i < len(labels) else f"Frame {i + 1}")

            c1, c2 = st.columns(2)

            with c1:
                st.image(
                    st.session_state.original_frames[i],
                    caption="Original",
                    use_container_width=True,
                )

            with c2:
                st.image(
                    st.session_state.enhanced_frames[i],
                    caption="Enhanced",
                    use_container_width=True,
                )

    st.header("Download Result")

    st.download_button(
        "Download Enhanced Video",
        data=st.session_state.download_bytes,
        file_name=f"output_{cvd_type}.mp4",
        mime="video/mp4",
    )
