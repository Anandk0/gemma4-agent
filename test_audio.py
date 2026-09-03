import torch
import torchaudio
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model.eval()

audio = "test.wav"

print("Loading:", audio)

wav, sr = torchaudio.load(audio)

print("Original shape:", wav.shape)
print("Sample rate:", sr)

# mono
wav = torch.mean(wav, dim=0, keepdim=True)

# 16 kHz
if sr != 16000:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=16000
    )
    wav = resampler(wav)

print("Final shape:", wav.shape)
print("Final sample rate: 16000")

with torch.no_grad():
    result = model(
        wav,
        "hi",
        "ctc"
    )

print("\nRESULT:")
print(repr(result))
0