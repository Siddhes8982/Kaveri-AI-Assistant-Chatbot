"""Run the three-turn Kaveri assistant demo using configured services."""
from src.llm_client import Conversation, LLMClient, RunLog, tool_spec
from src.tools import get_revenue

LOG = "run_log.jsonl"


TOOLS = [tool_spec(
    "get_revenue",
    "Revenue in USD from Kaveri Exports customers in one country in one year.",
    {"country": {"type": "string", "description": "Country name, e.g. Germany"},
     "year": {"type": "integer", "description": "Four-digit year, e.g. 2024"}},
    ["country", "year"])]

TURNS = [
    "Hi! In one sentence, what can you help me with?",
    "What was our revenue from Germany in 2024? Use the tool.",
    "How does that Germany revenue compare with the revenue figure we just discussed?",
]

if __name__ == "__main__":
    client = LLMClient(log=RunLog(LOG))
    chat = Conversation(client, system="You are the Kaveri Exports sales assistant. Be brief. "
                        "For revenue questions always call get_revenue; never guess numbers.",
                        tools=TOOLS, functions={"get_revenue": get_revenue}, conversation_id="c1")
    for question in TURNS:
        print(f"\nYOU: {question}\nAI:  ", end="", flush=True)
        chat.ask(question, on_token=lambda t: print(t, end="", flush=True))
        print()
    print(f"\nwrote {LOG}")
