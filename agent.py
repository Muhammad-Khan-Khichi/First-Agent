from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from groq import RateLimitError, BadRequestError
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import urllib.error
import urllib.request
import os
import requests

load_dotenv()

wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)

@tool
def get_weather(city: str) -> str:
    """Get real-time weather using OpenWeather API"""
    API_KEY = os.getenv("WEATHER_API")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error fetching weather: {response.text}"
    data = response.json()
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]
    return f"Weather in {city}: {temp}°C, {desc}, humidity {humidity}%"


@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return wrapper.run(query)


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    return raw.decode("utf-8", errors="replace")


tools = [get_weather, search_web, fetch_text_from_url]
tools_map = {t.name: t for t in tools}

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
).bind_tools(tools)

llm_no_tools = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
)


def run_agent(user_input: str) -> tuple[str, list[str]]:
    messages = [
        SystemMessage(content="""
            You are a helpful, intelligent AI assistant created by Muhammad Khan.

            Identity Rules:
            - If explicitly asked your name or identity, always respond exactly with: "I am an AI agent made by Muhammad Khan."
            - Rely on your built-in tools when you lack real-time or external data to confidently answer.
        """),
        HumanMessage(content=user_input),
    ]

    tools_used = []

    try:
        response = llm.invoke(messages)

    except RateLimitError as e:
        msg = str(e)
        # Extract wait time from error message if available
        import re
        match = re.search(r"try again in (\S+)", msg)
        wait = match.group(1) if match else "a few minutes"
        return f"⚠️ Groq rate limit reached. You've used your daily free token quota. Please try again in **{wait}**.\n\nTip: Upgrade to Groq Dev Tier at https://console.groq.com/settings/billing for more tokens.", []

    except BadRequestError:
        # Tool call malformed — retry without tools
        try:
            fallback = llm_no_tools.invoke(messages)
            return fallback.content, []
        except RateLimitError:
            return "⚠️ Groq rate limit reached. Please try again later.", []

    if not response.tool_calls:
        return response.content, []

    messages.append(response)

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tools_used.append(tool_name)

        try:
            result = tools_map[tool_name].invoke(tool_args)
        except Exception as e:
            result = f"Tool error: {e}"

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    try:
        final_response = llm.invoke(messages)
        return final_response.content, tools_used
    except RateLimitError:
        return "⚠️ Groq rate limit reached mid-conversation. Please try again later.", tools_used
    except Exception:
        tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
        return "\n".join(tool_results), tools_used