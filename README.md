# Boxbnb Staff: Custom ReAct Agents with Configurable Personas

EECE 798S, C4: Agent Personas & Thinking Mechanisms.

Boxbnb is a fictional short-stay rental platform for cats. Three of its staff members are each a ReAct agent, with
their own job, persona, tools and LLM configuration. All three run on one hand-written ReAct loop, built as a
LangGraph state machine.

| Agent | Job | Tools | Config |
|---|---|---|---|
| Bob the dog | Customer support: questions, bookings, complaints | `lookup_listing`, `record_booking`, `record_feedback` | balanced |
| Ivy-BeanBun | Plans a stay within a cat's treat budget | `search_listings`, `human_schedule`, `get_quote` | playful |
| Ashe Burrito | Certifies (or rejects) spots hosts submit | `certification_criteria`, `score_spot`, `record_certification` | strict |

## What's here

- `boxbnb_staff.ipynb`: the whole assignment. It covers the use case, the ReAct loop (reason, then act, then observe,
  plus a node that repairs bad formatting), the personas and instructions, the experiments, and the reflection.
- `app.py`: the three agents as a Gradio chat app. You pick who to talk to, and each reply shows its ReAct trace.
- `experiments/`: every experiment run with its full trace (`runs.jsonl`), the summary table, and the voice ratings.
- `me/business_summary.txt` and `data/`: the business docs, the certification checklist, and the human's weekly
  schedule.
- `probe_ollama.py`: the check that showed which sampling settings Ollama actually applies.

## The ReAct loop

The model writes plain text: `Thought:`, then either `Action: tool_name {json}` or `Final Answer:`. Generation stops
at `Observation:`. Our code parses the action, runs the tool, appends the result as the Observation, and loops back to
the model. No framework executor and no built-in function calling.

## Run it

Needs [Ollama](https://ollama.com) running, with `llama3.1:8b` pulled. In PowerShell, from the repo folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # optional: a Gemini key, only used in the model-type comparison
python app.py
```

Then open http://127.0.0.1:7860. The notebook also runs top to bottom. Its experiment runner skips any run already
logged in `experiments/runs.jsonl`.
