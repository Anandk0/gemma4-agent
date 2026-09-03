import torch
import torchaudio
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

print("=" * 60)
print("Loading IndicConformer")
print("=" * 60)

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("\nModel loaded successfully!")
print("Model type:", type(model))

print("\nTesting CUDA...")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nREADY")