import os
import io
import json
import torch
import soundfile as sf
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# --------------------------------------------------
# Environment
# --------------------------------------------------

st.set_page_config(
    page_title="IndicConformer ASR Test",
    page_icon="🎙️",
)

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

# --------------------------------------------------
# CUDA libraries
# --------------------------------------------------

os.environ["LD_LIBRARY_PATH"] = (
    "/opt/conda/envs/gemma4/lib/python3.10/site-packages/nvidia/cudnn/lib:"
    "/opt/conda/envs/gemma4/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
    "/opt/conda/envs/gemma4/lib/python3.10/site-packages/nvidia/cublas/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    from transformers import AutoModel

    st.write("Loading IndicConformer...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model.eval()

    return model


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🎙️ IndicConformer ASR Test")

st.write(
    "Record your voice and test IndicConformer transcription."
)

model = load_model()

st.success("ASR model loaded.")

st.write("### Record your voice")

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    use_container_width=True,
    format="wav",
    key="recorder",
)

# --------------------------------------------------
# Process audio
# --------------------------------------------------

if audio:

    st.audio(audio["bytes"], format="audio/wav")

    st.write("### Processing...")

    audio_bytes = audio["bytes"]

    # Read WAV
    waveform, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Convert stereo → mono
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    st.write(f"Sample rate: {sample_rate} Hz")
    st.write(f"Duration: {len(waveform) / sample_rate:.2f} sec")

    # Convert to torch
    wav = torch.tensor(waveform).unsqueeze(0)

    # IndicConformer expects audio at 16 kHz
    if sample_rate != 16000:

        import torchaudio

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )

        wav = resampler(wav)

    # --------------------------------------------------
    # Select language
    # --------------------------------------------------

    language = st.selectbox(
        "Language",
        [
            ("Hindi", "hi"),
            ("English", "en"),
            ("Kannada", "kn"),
            ("Marathi", "mr"),
            ("Telugu", "te"),
            ("Tamil", "ta"),
            ("Gujarati", "gu"),
            ("Bengali", "bn"),
            ("Malayalam", "ml"),
            ("Odia", "or"),
            ("Punjabi", "pa"),
            ("Urdu", "ur"),
        ],
        format_func=lambda x: x[0]
    )[1]

    # --------------------------------------------------
    # Transcribe
    # --------------------------------------------------

    if st.button("📝 Transcribe", use_container_width=True):

        with st.spinner("Transcribing..."):

            with torch.no_grad():

                result = model(
                    wav,
                    lang=language,
                    decoding="rnnt"
                )

        st.write("### Transcript")

        st.success(result)