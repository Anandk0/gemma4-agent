import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration


# ============================================================
# CONFIG
# ============================================================

MODEL = "ai4bharat/indic-parler-tts"
OUTPUT_FILE = "test_tts.wav"

DEVICE = "cuda"
DTYPE = torch.float16


# ============================================================
# CHECK GPU
# ============================================================

print("=" * 60)
print("INDIC PARLER TTS TEST")
print("=" * 60)

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Indic Parler TTS...")

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
)

model = model.to(DEVICE)

print("TTS model loaded successfully.")
print("Model device:", next(model.parameters()).device)


# ============================================================
# LOAD TOKENIZERS
# ============================================================

print("\nLoading tokenizers...")

description_tokenizer = AutoTokenizer.from_pretrained(MODEL)
prompt_tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Tokenizers loaded.")


# ============================================================
# VOICE DESCRIPTION
# ============================================================

description = (
    "A female Indian speaker speaks clearly and naturally in Hindi. "
    "The voice is warm, friendly, professional and suitable for a "
    "telephone admission counselor. "
    "The speaker talks at a natural conversational pace with clear "
    "pronunciation."
)


# ============================================================
# TEXT TO SPEAK
# ============================================================

text = (
    "Namaste, Jain College of Engineering and Research mein "
    "aapka swagat hai."
)


print("\nText:")
print(text)

print("\nVoice description:")
print(description)


# ============================================================
# TOKENIZE
# ============================================================

print("\nTokenizing...")

description_input = description_tokenizer(
    description,
    return_tensors="pt",
)

prompt_input = prompt_tokenizer(
    text,
    return_tensors="pt",
)


# Move tensors to GPU
description_input = {
    key: value.to(DEVICE)
    for key, value in description_input.items()
}

prompt_input = {
    key: value.to(DEVICE)
    for key, value in prompt_input.items()
}


# ============================================================
# GENERATE SPEECH
# ============================================================

print("\nGenerating speech...")

with torch.inference_mode():

    generation = model.generate(
        input_ids=description_input["input_ids"],
        attention_mask=description_input["attention_mask"],
        prompt_input_ids=prompt_input["input_ids"],
        prompt_attention_mask=prompt_input["attention_mask"],
    )


# ============================================================
# CONVERT AUDIO
# ============================================================

print("\nGeneration complete.")

print("Original tensor:")
print("  dtype:", generation.dtype)
print("  shape:", generation.shape)
print("  device:", generation.device)


# IMPORTANT:
# soundfile does not support float16.
# Convert generated audio to float32 before saving.

audio = (
    generation
    .float()
    .cpu()
    .numpy()
    .squeeze()
)


print("\nAudio:")
print("  dtype:", audio.dtype)
print("  shape:", audio.shape)
print("  samples:", len(audio))


# ============================================================
# SAVE WAV
# ============================================================

sample_rate = model.config.sampling_rate

print("\nSaving WAV...")
print("Sample rate:", sample_rate)

sf.write(
    OUTPUT_FILE,
    audio,
    sample_rate,
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("Audio saved to:")
print(OUTPUT_FILE)

print("File format: WAV")
print("Sample rate:", sample_rate)
print("Audio dtype:", audio.dtype)
print("Audio samples:", len(audio))

print("=" * 60)