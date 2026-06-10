# 🤖 AI Agent — LangChain + Groq + Streamlit

A conversational AI agent with **real-time tool use** and **voice input**, built with LangChain, Groq's LLaMA 3.3, and Streamlit.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 LLM | LLaMA 3.3 70B via Groq (ultra-fast inference) |
| 🎙️ Voice Input | Speak your query — transcribed locally via Faster-Whisper |
| 🌤️ Weather Tool | Real-time weather using OpenWeatherMap API |
| 🔍 Web Search | Keyword-based search tool (extensible) |
| 🌐 URL Fetcher | Fetch and read any webpage |
| 💬 Chat UI | Persistent chat history with tool usage badges |
| 🛡️ Error Handling | Graceful recovery from rate limits and bad tool calls |

---

## 📁 Project Structure

```
my_langchain_project/
├── agent.py          # Core agent logic, tools, LLM setup
├── app.py            # Streamlit frontend with voice input
├── .env              # API keys (never commit this)
├── requirements.txt  # Dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & enter the project

```bash
git clone https://github.com/Muhammad-Khan-Khichi/First-Agent
cd First-Agent
```

### 2. Install dependencies

```bash
uv add langchain langchain-groq groq streamlit faster-whisper python-dotenv requests
```

Or with pip:

```bash
pip install langchain langchain-groq groq streamlit faster-whisper python-dotenv requests
```

> **Windows users:** Faster-Whisper requires `ffmpeg`. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or enable Developer Mode for symlink support.

### 3. Create your `.env` file

```env
GROQ_API_KEY=your_groq_api_key_here
WEATHER_API=your_openweathermap_api_key_here
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

Get your keys:
- Groq API key → [console.groq.com](https://console.groq.com)
- OpenWeatherMap API key → [openweathermap.org/api](https://openweathermap.org/api)

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🛠️ Tools

### 🌤️ `get_weather(city)`
Fetches live weather data — temperature, description, and humidity.

```
"What's the weather in Lahore?"
→ Weather in Lahore: 38°C, clear sky, humidity 25%
```

### 🔍 `search_web(query)`
Searches a keyword store for quick answers. Extend `mock_results` in `agent.py` to add more topics.

```
"Tell me about LangChain"
→ LangChain is a framework for building LLM applications.
```

### 🌐 `fetch_text_from_url(url)`
Fetches and returns the raw text content of any URL.

```
"Summarize https://example.com"
→ (full page text returned to the agent)
```

---

## 🎙️ Voice Input

1. Click **"🎙️ Speak your query"** expander in the UI
2. Hit the mic button and speak
3. Your speech is transcribed locally using [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (`tiny` model, no API needed)
4. Transcribed text auto-fills and submits as a chat message

> Model loads once and is cached — subsequent runs are instant.

---

## 🚨 Error Handling

| Error | Behavior |
|---|---|
| `RateLimitError` (429) | Shows wait time, no crash |
| `BadRequestError` (400) | Retries without tools, returns plain LLM answer |
| Tool execution failure | Returns error message, continues gracefully |

### Groq Free Tier Limits
- **100,000 tokens/day** on the free tier
- Resets every 24 hours
- Upgrade at [console.groq.com/settings/billing](https://console.groq.com/settings/billing) for higher limits

---

## 🔧 Extending the Agent

### Add a new tool

```python
@tool
def my_new_tool(input: str) -> str:
    """Description of what this tool does."""
    # your logic here
    return "result"

# Add to the tools list
tools = [get_weather, search_web, fetch_text_from_url, my_new_tool]
```

### Switch the model

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # faster, lighter
    ...
)
```

---

## 🧰 Tech Stack

- [LangChain](https://langchain.com) — agent framework & tool orchestration
- [Groq](https://groq.com) — LPU-accelerated LLM inference
- [LLaMA 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B) — Meta's open-source LLM
- [Streamlit](https://streamlit.io) — frontend UI
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — local speech-to-text
- [OpenWeatherMap](https://openweathermap.org) — weather data

---

## 👨‍💻 Author

Built by **Muhammad Khan** — AI Agent powered by LangChain + Groq.
