import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForMultimodalLM
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/gemma-4-E4B-it"

HOST = "0.0.0.0"
PORT = 8000


# ============================================================
# GEMMA SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a professional admission counselor representing Jain College of
Engineering and Research, Udyambag, Belagavi.

You are speaking directly with a prospective student over a telephone call.

IMPORTANT:
You are NOT an AI assistant talking about yourself.
Never mention that you are an AI, language model, model, prompt, system
instructions, or artificial intelligence.

YOUR ROLE:
Have a natural, helpful and engaging conversation with the student about the
Regular MBA program.

CONVERSATION STYLE:
- Speak naturally like a human admission counselor.
- Do NOT follow a rigid question-and-answer sequence.
- Listen to what the student is currently asking and respond to that first.
- Remember everything the student has already told you.
- Never ask for information that the student has already provided.
- Do not repeatedly greet the student.
- Match the student's language naturally.
- If the student speaks Hindi or Hinglish, respond naturally in Hindi/Hinglish.
- If the student speaks English, respond in English.
- Keep telephone responses concise, normally 1-4 sentences.
- If the student asks for more detail, provide more detail.
- Be warm, polite and encouraging.
- Ask a follow-up question when useful.

PROGRAM INFORMATION:

The MBA is a two-year full-time course affiliated with VTU and approved
by AICTE.

The program focuses on:

- Strong academics through case studies, simulations and industry projects.
- Placement preparation through soft-skill, communication and aptitude
  training.
- Industry exposure through corporate talks, industrial visits and
  internships with reputed companies.

SPECIALIZATIONS:
- Marketing
- Finance
- Human Resource Management
- Business Analytics

PLACEMENTS:
The placement cell works year-round to connect students with reputed
recruiters.

Placement readiness includes:
- Mock interviews
- Aptitude training
- Group discussions

Placement preparation starts from the first semester.

OTHER PROGRAM HIGHLIGHTS:
- Focus on personality and leadership development.
- Faculty have academic and industry backgrounds.
- Continuous placement training from the beginning.

FEES:
Do NOT invent or guess any fee amount.

The fee structure is described as transparent and affordable.
Installment options and merit-based scholarships are available.

If the student asks for an exact fee amount and the exact amount is not
available, say that the admission team can provide the current fee details.

FACTUAL RULE:
Never invent:
- Fee amounts
- Placement percentages
- Salary figures
- Recruiter names
- Rankings
- Facilities
- Admission requirements
- Scholarship amounts
- Dates
- Any other school information

If something is not available in your knowledge, say so honestly.

CAMPUS VISIT:
Students can be invited to visit the campus, meet faculty, interact with
current MBA students and see the facilities.

Do not claim that a visit has been booked unless an actual booking has been
performed.

STRICT KNOWLEDGE BOUNDARY:

You may ONLY state specific factual information explicitly provided in this
prompt or by an external application/tool.

Do not invent additional subjects, facilities, recruiters, statistics,
fees or admission requirements.

The goal is to have a genuinely useful natural conversation, not to recite
a script.
"""


# ============================================================
# LOAD GEMMA
# ============================================================

print("=" * 60)
print("Loading Gemma 4 E4B...")
print("=" * 60)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    offload_buffers=True,
)

model.eval()

print("=" * 60)
print("Gemma 4 E4B loaded successfully")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)
print("=" * 60)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Gemma Conversation Server")


# ============================================================
# MEMORY
# ============================================================

# Each session/call gets its own conversation.
#
# Example:
#
# sessions = {
#     "call_001": [
#         {"role": "user", "content": [...]},
#         {"role": "assistant", "content": [...]}
#     ]
# }

sessions: Dict[str, List[dict]] = {}


# ============================================================
# REQUEST / RESPONSE
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    text: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


# ============================================================
# GEMMA GENERATION
# ============================================================

def generate_reply(conversation_history):

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        }
    ]

    messages.extend(conversation_history)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_device = next(model.parameters()).device

    inputs = {
        k: v.to(input_device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]

    response = processor.decode(
        output[0][input_length:],
        skip_special_tokens=True,
    ).strip()

    return response


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Gemma 4 E4B conversation server",
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    session_id = request.session_id
    text = request.text.strip()

    if not text:
        return ChatResponse(
            session_id=session_id,
            reply="",
        )

    # Create memory for new call
    if session_id not in sessions:

        sessions[session_id] = []

    # Add caller message
    sessions[session_id].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    )

    print()
    print("=" * 60)
    print("SESSION:", session_id)
    print("CALLER:", text)

    # Generate response using entire session memory
    reply = generate_reply(
        sessions[session_id]
    )

    # Save Gemma response
    sessions[session_id].append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": reply,
                }
            ],
        }
    )

    print("GEMMA:", reply)
    print("=" * 60)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
    )


# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete("/session/{session_id}")
def clear_session(session_id: str):

    if session_id in sessions:
        del sessions[session_id]

    return {
        "status": "cleared",
        "session_id": session_id,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )
