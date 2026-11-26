# Kaveri AI Assistant: reusable LLM client

This workspace contains the reusable OpenAI-compatible client, conversation/tool support,
and a three-turn streaming demo. It targets the sales schema documented in
`business_brief.md`: revenue is `SUM(orders.total_amount)` joined to `customers` by
`customer_id`, filtered by country and `order_date`.

## Configure

Install dependencies with `python -m pip install -r requirements.txt`, then copy
`.env.example` to `.env` and set `LLM_API_KEY` and the existing `DATABASE_URL` locally.
The demo uses `LLM_BASE_URL=https://openrouter.ai/api/v1` and
`LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free` by default in the example file.
Never commit `.env` or put credentials in source code.

## Run

Run `pytest -v` for offline tests and `python chat_demo.py` for the live three-turn demo.
The demo appends events to `run_log.jsonl`. A real database and provider key must be
configured; the program does not generate sample revenue or responses.
