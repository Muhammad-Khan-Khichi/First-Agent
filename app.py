import os
import tempfile
import streamlit as st
from faster_whisper import WhisperModel
from agent import run_agent

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)


@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )

with st.spinner("Loading speech model..."):
    whisper_model = load_whisper_model()

st.title("🤖 AI Agent")
st.write("Chat with your LangChain + Groq agent using text or voice 🎙️")

if "chat" not in st.session_state:
    st.session_state.chat = []

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

audio_file = st.audio_input("🎙️ Record your message")

if audio_file:
    with st.spinner("Transcribing..."):
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        segments, _ = whisper_model.transcribe(tmp_path)

        transcribed = " ".join(
            segment.text for segment in segments
        ).strip()

        os.unlink(tmp_path)

    if transcribed:
        st.success(f"🎤 {transcribed}")
        st.session_state.pending_input = transcribed


typed_input = st.chat_input("Ask something...")


user_input = None

if typed_input:
    user_input = typed_input

elif st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None


if user_input:


    st.session_state.chat.append(("user", user_input))


    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            answer, tools_used = run_agent(user_input)

        if tools_used:
            tools_str = " · ".join(
                f"`{tool}`" for tool in tools_used
            )

            st.markdown(
                f"<small style='color:gray'>🔧 Used: {tools_str}</small>",
                unsafe_allow_html=True
            )

        st.markdown(answer)


    st.session_state.chat.append(("assistant", answer))