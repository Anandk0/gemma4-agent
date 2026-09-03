import os
import io
import requests
import torch
import streamlit as st

from transformers import AutoProcessor, AutoModelForMultimodalLM


# ============================================================
# CONFIG
# ============================================================

GEMMA_MODEL_ID = os.environ.get(
    "GEMMA_MODEL_ID",
    "google/gemma-4-E4B-it"
)

# Your Cloudflare URL for the ASR server
ASR_URL = os.environ.get(
    "ASR_URL",
    "https://compared-provides-furniture-saw.trycloudflare.com"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a professional admission counselor representing Jain College of
Engineering and Research, Udyambag, Belagavi.

You are speaking directly with a prospective student over a telephone call.

You are NOT an AI assistant talking about yourself.

Never mention:
- AI
- language model
- model
- system prompt
- system instructions
- artificial intelligence

ROLE:

Have a natural conversation with the student about the Regular MBA program.

CONVERSATION STYLE:

- Speak naturally like a human admission counselor.
- Listen to the student's current question.
- Answer the current question first.
- Remember information already provided by the student.
- Never ask again for information already provided.
- Do not repeatedly greet the student.
- If the student speaks Hindi or Hinglish, respond naturally in Hindi/Hinglish.
- If the student speaks English, respond in English.
- Keep responses concise.
- Normally use 1-4 sentences.
- Be warm, polite and helpful.
- Ask a follow-up question when useful.
- Do not force a rigid question sequence.

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

- Personality and leadership development.
- Faculty have academic and industry backgrounds.
- Continuous placement training from the beginning.

FEES:

Do NOT invent or guess any fee amount.

The fee structure is described as transparent and affordable.
Installment options and merit-based scholarships are available.

If the student asks for the exact fee and you do not know the exact
current amount, say that the admission team can provide the current
fee details.

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

If something is not available, say so honestly.

CAMPUS VISIT:

Students can be invited to visit the campus, meet faculty, interact
with current MBA students and see the facilities.

Do not claim that a visit has been booked unless an actual booking
has been performed.

STRICT KNOWLEDGE BOUNDARY:

Only state specific factual information explicitly provided in this prompt
or provided by an external application/tool.

Do not expand topics using general knowledge.

The goal is a genuinely useful natural telephone conversation, not a
script being recited.
"""


# ============================================================
# LOAD GEMMA
# ============================================================

@st.cache_resource
def load_gemma():

    st.info("Loading Gemma 4 E4B...")

    processor = AutoProcessor.from_pretrained(
        GEMMA_MODEL_ID
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        offload_buffers=True,
    )

    st.success("Gemma 4 E4B loaded successfully.")

    return processor, model


# ============================================================
# ASR
# ============================================================

def transcribe_audio(audio_bytes, filename="audio.wav", language="kn"):

    url = ASR_URL.rstrip("/") + "/transcribe"

    files = {
        "file": (
            filename,
            io.BytesIO(audio_bytes),
            "audio/wav"
        )
    }

    data = {
        "language": language,
        "decoding": "ctc",
    }

    try:

        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=120,
        )

    except requests.exceptions.RequestException as e:

        return {
            "status": "error",
            "error": f"Could not connect to ASR server: {e}",
        }

    if response.status_code != 200:

        return {
            "status": "error",
            "error": (
                f"ASR server returned HTTP "
                f"{response.status_code}: {response.text}"
            ),
        }

    try:

        result = response.json()

    except Exception:

        return {
            "status": "error",
            "error": "ASR server returned invalid JSON.",
            "raw_response": response.text,
        }

    return result


# ============================================================
# GEMMA RESPONSE
# ============================================================

def generate_reply(
    processor,
    model,
    conversation_history
):

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
            max_new_tokens=100,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]

    response = processor.decode(
        output[0][input_length:],
        skip_special_tokens=True,
    ).strip()

    return response


# ============================================================
# LOAD GEMMA
# ============================================================

processor, gemma_model = load_gemma()


# ============================================================
# SESSION MEMORY
# ============================================================

if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


# ============================================================
# UI
# ============================================================

st.title("Jain College MBA Calling Agent")

st.caption(
    "Gemma 4 E4B + Indic Conformer ASR"
)


# ============================================================
# ASR SERVER STATUS
# ============================================================

with st.sidebar:

    st.header("Configuration")

    st.write("Gemma:")
    st.code(GEMMA_MODEL_ID)

    st.write("ASR Server:")
    st.code(ASR_URL)

    if st.button("Test ASR connection"):

        try:

            response = requests.get(
                ASR_URL.rstrip("/") + "/health",
                timeout=10,
            )

            if response.ok:

                st.success(
                    "ASR server connected!"
                )

                st.json(response.json())

            else:

                st.error(
                    f"ASR returned HTTP {response.status_code}"
                )

        except Exception as e:

            st.error(
                f"ASR connection failed: {e}"
            )


# ============================================================
# DISPLAY CONVERSATION
# ============================================================

for message in st.session_state.conversation_history:

    role = message["role"]

    text = message["content"][0]["text"]

    if role == "user":

        with st.chat_message("user"):
            st.write(text)

    elif role == "assistant":

        with st.chat_message("assistant"):
            st.write(text)


# ============================================================
# AUDIO INPUT
# ============================================================

st.subheader("Speak to the admission counselor")

audio = st.audio_input(
    "Click here and speak"
)


# ============================================================
# PROCESS AUDIO
# ============================================================

if audio is not None:

    audio_bytes = audio.getvalue()

    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    if st.button(
        "Transcribe and Ask",
        type="primary"
    ):

        # ----------------------------------------------------
        # ASR
        # ----------------------------------------------------

        with st.spinner(
            "Sending audio to Indic Conformer..."
        ):

            asr_result = transcribe_audio(
                audio_bytes,
                filename="caller.wav",
                language="kn",
            )

        # ----------------------------------------------------
        # ASR ERROR
        # ----------------------------------------------------

        if asr_result.get("status") != "success":

            st.error(
                "ASR failed"
            )

            st.json(asr_result)

        else:

            user_text = asr_result.get(
                "text",
                ""
            ).strip()

            # ------------------------------------------------
            # SHOW TRANSCRIPTION
            # ------------------------------------------------

            st.info(
                f"Caller: {user_text}"
            )

            if not user_text:

                st.warning(
                    "No speech was detected."
                )

            else:

                # --------------------------------------------
                # ADD USER MESSAGE
                # --------------------------------------------

                st.session_state.conversation_history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_text,
                            }
                        ],
                    }
                )

                # --------------------------------------------
                # GEMMA
                # --------------------------------------------

                with st.spinner(
                    "Gemma is thinking..."
                ):

                    reply = generate_reply(
                        processor,
                        gemma_model,
                        st.session_state.conversation_history,
                    )

                # --------------------------------------------
                # ADD ASSISTANT RESPONSE
                # --------------------------------------------

                st.session_state.conversation_history.append(
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

                st.rerun()


# ============================================================
# TEXT INPUT
# ============================================================

st.divider()

st.subheader("Or type a message")

user_text = st.chat_input(
    "Type what the caller says..."
)


if user_text:

    st.session_state.conversation_history.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                }
            ],
        }
    )

    with st.spinner(
        "Gemma is thinking..."
    ):

        reply = generate_reply(
            processor,
            gemma_model,
            st.session_state.conversation_history,
        )

    st.session_state.conversation_history.append(
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

    st.rerun()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.divider()

    if st.button("Clear conversation"):

        st.session_state.conversation_history = []

        st.rerun()

    st.write("GPU:")

    st.write(
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    if torch.cuda.is_available():

        st.write(
            "GPU memory:",
            round(
                torch.cuda.memory_allocated() / 1024**3,
                2
            ),
            "GB"
        )