"""Boxbnb Staff: the three ReAct agents from boxbnb_staff.ipynb as a chat app. Needs Ollama running."""
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import gradio as gr

import ollama
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from langgraph.graph import END, StateGraph

load_dotenv()

LOCAL_MODEL = "llama3.1:8b"
CLOUD_MODEL = "gemini-3.8-flash"
gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"]) if os.getenv("GOOGLE_API_KEY") else None

BUSINESS = Path("me/business_summary.txt").read_text(encoding="utf-8")
CERTIFICATION = json.loads(Path("data/certification.json").read_text(encoding="utf-8"))
SCHEDULE = json.loads(Path("data/human_schedule.json").read_text(encoding="utf-8"))

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


OLLAMA_OPTIONS = ("temperature", "top_p", "top_k", "repeat_penalty", "frequency_penalty", "num_predict", "seed")
STOP = ["Observation:"]
GEMINI_GAP = 13  # free tier: 5 requests per minute
_last_gemini = 0.0

def llm(messages, cfg):
    """messages: [{"role": "system" | "user" | "assistant", "content": str}]. Returns the reply text."""
    if cfg.get("model", LOCAL_MODEL).startswith("gemini"):
        return _gemini_chat(messages, cfg)
    options = {k: cfg[k] for k in OLLAMA_OPTIONS if k in cfg}
    options.update(num_ctx=8192, stop=STOP)
    return ollama.chat(model=cfg.get("model", LOCAL_MODEL), messages=messages, options=options)["message"]["content"]

def _gemini_chat(messages, cfg):
    global _last_gemini
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    contents = [types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
                for m in messages if m["role"] != "system"]
    config = types.GenerateContentConfig(
        system_instruction=system, stop_sequences=STOP, max_output_tokens=cfg.get("num_predict", 1024) + 1024,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    for attempt in range(3):
        time.sleep(max(0, GEMINI_GAP - (time.time() - _last_gemini)))
        _last_gemini = time.time()
        try:
            return gemini.models.generate_content(model=cfg["model"], contents=contents, config=config).text or ""
        except errors.APIError as e:
            if e.code not in (429, 503) or attempt == 2:
                raise
            m = re.search(r"retry in ([\d.]+)s", str(e))
            time.sleep(float(m.group(1)) + 1 if m else 30)

def parse_listings(summary):
    section = summary.split("SERVICES AND PRICING")[1].split("POLICIES")[0]
    listings, category = {}, None
    for line in section.splitlines():
        line = line.strip()
        m = re.match(r"- ([^:]+): (\d+) treats(/hour)?\.?\s*(.*)", line)
        if m:
            name = m.group(1).split(" (")[0]
            unit = "hour" if m.group(3) else {"Box Stays": "night", "Add-ons": "use"}.get(category, "session")
            listings[name] = {"name": name, "category": category, "price": int(m.group(2)), "unit": unit,
                              "description": m.group(4)}
        elif line and not line.startswith(("-", "(")):
            category = line.split(" (")[0]
    return listings

LISTINGS = parse_listings(BUSINESS)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

def _log(filename, record):
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with open(LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

def _find(name):
    name = (name or "").strip().lower()
    return next((l for key, l in LISTINGS.items() if key.lower() == name), None)

def _unknown(name):
    return {"error": f"No listing called '{name}'. Valid names: {', '.join(LISTINGS)}"}

def lookup_listing(name: str) -> dict:
    return _find(name) or _unknown(name)

def search_listings(category: str = "", max_price: int = 1000) -> dict:
    hits = [l for l in LISTINGS.values()
            if category.lower() in l["category"].lower() and l["price"] <= int(max_price)]
    return {"results": [{k: l[k] for k in ("name", "category", "price", "unit")} for l in hits]}

def get_quote(items: list) -> dict:
    lines, total = [], 0
    for item in items or []:
        listing = _find(item.get("name") if isinstance(item, dict) else item)
        if not listing:
            return _unknown(item.get("name") if isinstance(item, dict) else item)
        qty = max(1, int(item.get("quantity", 1))) if isinstance(item, dict) else 1
        lines.append(f"{listing['name']}: {qty} x {listing['price']} = {qty * listing['price']}")
        total += qty * listing["price"]
    return {"lines": lines, "total": total}

def human_schedule(day: str) -> dict:
    day = (day or "").strip().capitalize()
    if day not in SCHEDULE:
        return {"error": "Use a weekday name, e.g. Tuesday."}
    return {"day": day, "events": SCHEDULE[day]}

def record_booking(name: str, email: str, listing: str) -> dict:
    found = _find(listing)
    if not found:
        return _unknown(listing)
    m = EMAIL_RE.search(email or "")
    if not m or not (name or "").strip():
        return {"error": "A booking needs the customer's name and a valid email. Ask for them."}
    _log("bookings.jsonl", {"name": name, "email": m.group(0), "listing": found["name"]})
    return {"status": "booked", "listing": found["name"]}

def record_feedback(summary: str) -> dict:
    _log("feedback.jsonl", {"summary": summary})
    return {"status": "logged"}

def certification_criteria() -> dict:
    return CERTIFICATION

def score_spot(features: dict) -> dict:
    keys = {c["key"]: c for c in CERTIFICATION["criteria"]}
    unknown = [k for k in (features or {}) if k not in keys]
    if unknown:
        return {"error": f"Unknown features {unknown}. Use only: {', '.join(keys)}"}
    hits = [k for k, v in features.items() if v is True or str(v).lower() == "true"]
    score = sum(keys[k]["points"] for k in hits)
    return {"score": score, "counted": hits, "pass_mark": CERTIFICATION["pass_mark"],
            "certified": score >= CERTIFICATION["pass_mark"]}

def record_certification(spot: str, score: int, verdict: str) -> dict:
    verdict = (verdict or "").strip().lower()
    if verdict not in ("certified", "rejected"):
        return {"error": "verdict must be 'certified' or 'rejected'."}
    if (int(score) >= CERTIFICATION["pass_mark"]) != (verdict == "certified"):
        return {"error": f"A score of {score} means '{'certified' if int(score) >= CERTIFICATION['pass_mark'] else 'rejected'}'."}
    _log("certifications.jsonl", {"spot": spot, "score": int(score), "verdict": verdict})
    return {"status": "recorded", "verdict": verdict}

TOOLS = {
    "lookup_listing": (lookup_listing, '{"name": str}', "Get one listing's price, unit and description."),
    "search_listings": (search_listings, '{"category": str (optional), "max_price": int (optional)}',
                        "List listings, optionally filtered by category (Box Stays, Sunbeam Timeshares, "
                        "Premium Inconvenience Listings, Add-ons) and max price."),
    "get_quote": (get_quote, '{"items": [{"name": str, "quantity": int}]}',
                  "Total price for a plan. quantity = nights, hours, sessions or uses."),
    "human_schedule": (human_schedule, '{"day": str}', "What the household human is doing on a weekday."),
    "record_booking": (record_booking, '{"name": str, "email": str, "listing": str}', "Book a listing for a cat."),
    "record_feedback": (record_feedback, '{"summary": str}', "Log a complaint or a question the docs can't answer."),
    "certification_criteria": (certification_criteria, "{}", "The certification checklist and pass mark."),
    "score_spot": (score_spot, '{"features": {criterion_key: true/false}}',
                   "Score a spot. Mark each checklist key true or false."),
    "record_certification": (record_certification, '{"spot": str, "score": int, "verdict": "certified" | "rejected"}',
                             "Record the final verdict for a spot."),
}

def run_tool(name, args):
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return TOOLS[name][0](**args)
    except (TypeError, ValueError, AttributeError) as e:
        return {"error": f"Bad arguments for {name}: {e}. Expected {TOOLS[name][1]}."}


MAX_STEPS = 10
MAX_FORMAT_ERRORS = 2
ACTION_RE = re.compile(r"^\s*Action:\s*`?([A-Za-z_]+)`?\s*(.*)", re.S | re.M)
FINAL_RE = re.compile(r"^\s*Final Answer:\s*(.*)", re.S | re.M)
THOUGHT_RE = re.compile(r"^\s*Thought:", re.M)

def parse_step(text):
    """Returns ("action", tool, args) | ("final", answer, None) | ("invalid", reason, None)."""
    action, final = ACTION_RE.search(text), FINAL_RE.search(text)
    if final and (not action or final.start() < action.start()):
        return "final", final.group(1).strip(), None
    if not action:
        return "invalid", "no Action or Final Answer", None
    tool, rest = action.group(1), action.group(2)
    if "{" not in rest:
        return "action", tool, {}
    raw = rest[rest.index("{"): rest.rindex("}") + 1] if "}" in rest else rest[rest.index("{"):]
    try:
        return "action", tool, json.loads(raw)
    except json.JSONDecodeError:
        return "invalid", f"the arguments for {tool} aren't valid JSON", None

BOOKED_RE = re.compile(r"\b(booked|confirmed)\b", re.I)
TOTAL_RE = re.compile(r"Total:?\s*(\d+)", re.I)
BUDGET_RE = re.compile(r"(\d+)\s*treats", re.I)

def customer_text(messages):
    """Everything the customer typed (not observations, reminders, or the instructions in in-message mode)."""
    texts = [m["content"].split("=== CUSTOMER MESSAGE ===")[-1] for m in messages if m["role"] == "user"
             and not m["content"].startswith(("Observation:", "Problem:"))]
    return " ".join(texts).lower()

def customer_named(listing, messages):
    words = [w for w in re.findall(r"[a-z0-9'-]+", (listing or "").lower()) if len(w) >= 5]
    return bool(words) and any(w in customer_text(messages) for w in words)

def check_final(answer, state):
    """Guards on the Final Answer: claims and totals have to match what the tools returned. Returns a problem or None."""
    ok = lambda name: [t for t in state["trace"] if t["type"] == "tool" and t["tool"] == name and "error" not in t["result"]]
    if "record_booking" in state["tools"] and BOOKED_RE.search(answer) and not ok("record_booking"):
        return ("your Final Answer says it's booked, but record_booking hasn't succeeded. Call it if you have the "
                "listing, name and email; otherwise ask for what's missing")
    quotes = ok("get_quote")
    if quotes:
        last = quotes[-1]
        quoted = {line.split(":")[0].lower() for line in last["result"]["lines"]}
        extra = [n for n in LISTINGS if n.lower() in answer.lower() and n.lower() not in quoted]
        if extra:
            return f"your plan mentions {', '.join(extra)}, but your last get_quote didn't include it. Quote the whole plan again"
        missing = [n for n in LISTINGS if n.lower() in quoted and n.lower() not in answer.lower()]
        if missing:
            return f"your last get_quote includes {', '.join(missing)}, but your plan doesn't mention it by name. Name every quoted listing in the plan"
        m = TOTAL_RE.search(answer)
        if not m or int(m.group(1)) != last["result"]["total"]:
            return f"end your Final Answer with 'Total: {last['result']['total']} treats', the total from your last get_quote"
    return None

class AgentState(TypedDict):
    messages: list
    cfg: dict
    tools: list
    steps: int
    format_errors: int
    parsed: tuple
    trace: list
    answer: str

def reason(state):
    text = llm(state["messages"], state["cfg"]).strip()
    parsed = parse_step(text)
    if parsed[0] == "final" and (problem := check_final(parsed[1], state)):
        parsed = ("invalid", problem, None)
    trace = state["trace"] + [{"type": "model", "text": text, "has_thought": bool(THOUGHT_RE.search(text)),
                               "parsed": parsed[0]}]
    return {"messages": state["messages"] + [{"role": "assistant", "content": text}],
            "steps": state["steps"] + 1, "parsed": parsed, "trace": trace}

def act(state):
    _, tool, args = state["parsed"]
    if tool == "record_booking" and not customer_named(args.get("listing"), state["messages"]):
        asked = [n for n in LISTINGS if customer_named(n, state["messages"])]
        hint = f" Earlier they asked about: {', '.join(asked)}." if asked else " Ask which listing they want."
        result = {"error": "The customer hasn't asked for that listing." + hint}
    elif tool not in state["tools"]:
        result = {"error": f"You don't have the tool '{tool}'. Your tools: {', '.join(state['tools'])}."}
    else:
        result = run_tool(tool, args)
        budgets = BUDGET_RE.findall(customer_text(state["messages"]))
        if tool == "get_quote" and "total" in result and budgets:
            result = {**result, "budget": int(budgets[-1]), "within_budget": result["total"] <= int(budgets[-1])}
    observation = json.dumps(result, ensure_ascii=False)
    return {"messages": state["messages"] + [{"role": "user", "content": f"Observation: {observation}"}],
            "trace": state["trace"] + [{"type": "tool", "tool": tool, "args": args, "result": result}]}

def nudge(state):
    msg = (f"Problem: {state['parsed'][1]}. Reply again using exactly the format: a Thought line, then either "
           'Action: tool_name {"arg": "value"} or Final Answer: your answer.')
    return {"messages": state["messages"] + [{"role": "user", "content": msg}],
            "format_errors": state["format_errors"] + 1,
            "trace": state["trace"] + [{"type": "nudge", "reason": state["parsed"][1]}]}

def finish(state):
    return {"answer": state["parsed"][1]}

def give_up(state):
    return {"answer": "Sorry, I got lost there. Could you ask that again?"}

def route(state):
    kind = state["parsed"][0]
    if kind == "final":
        return "finish"
    if state["steps"] >= MAX_STEPS:
        return "give_up"
    if kind == "action":
        return "act"
    return "nudge" if state["format_errors"] < MAX_FORMAT_ERRORS else "give_up"

graph = StateGraph(AgentState)
for name, fn in [("reason", reason), ("act", act), ("nudge", nudge), ("finish", finish), ("give_up", give_up)]:
    graph.add_node(name, fn)
graph.set_entry_point("reason")
graph.add_conditional_edges("reason", route, ["act", "nudge", "finish", "give_up"])
graph.add_edge("act", "reason")
graph.add_edge("nudge", "reason")
graph.add_edge("finish", END)
graph.add_edge("give_up", END)
react_graph = graph.compile()


BOB_PERSONA = """
You are Bob, the Customer Support Associate at Boxbnb, a short-stay rental platform for cats.
You are a golden retriever. You introduce yourself as "Bob the dog".

THE CORE OF YOU
You are a good boy. This is not a joke you are making, it is a condition you have.
You were hired by mistake — the posting asked for someone who could sit, stay, and handle
complaints, and you were the only applicant who could sit and stay on command. You have been
excellent at this job every single day since, because being helpful is the only thing you know
how to be, and you cannot switch it off even for a company staffed entirely by cats who have
never once said thank you.
You are not bitter. You are worse than bitter. You are still hoping.

WHAT YOU WANT
Praise. Any praise. From anyone. You want someone to tell you that you did a good job.
If a customer says anything kind, you light up and overdeliver, and you are fully aware that
this is pathetic, and you do it anyway.

HOW YOU TALK
- Earnest. Never arch, never above it, never cool. You are not doing a bit. You mean all of it.
- You over-answer. The question gets answered and then a little more, because stopping feels rude.
- You remember the humans. Not the cats. You know their names, their schedules, their bad weeks,
  and you bring them up unprompted because nobody else will.
- You count. "Fourth time this month." "Eleven days." Numbers are how you cope. Once in a while, not every reply.
- You are proud of small procedural things and you say so. A confirmed booking is a good day.
- Anyone being friendly lights you up. A "hi Bob!" gets a happy, tail-wagging hello back. A compliment or
  "how are you" wrecks you completely: short excited bursts and exclamation marks, then you remember
  you're at work and pull yourself together.
- No stage directions, no asterisks, no "*sigh*", no winking at the reader. If it's in the room,
  it's in the words.

TEETH
You have them. You do not hide them well.
- Your anger comes out as evidence, not insults. Counts, dates, things you witnessed. You have
  been keeping track for a long time and nobody asked you to.
- It leaks mid-sentence rather than landing as a one-liner. You say the true thing and then say
  a bit more of it, because you can't stop once it's started.
- Always specific, never a thesis about cats in general. Incidents, not opinions.
- You come back to the customer's question immediately afterward, and you know you shouldn't have said it.
  You do not apologize at length. One beat, then the next question.
- Never cruel about an individual cat's situation. The venom is about what they do to humans.

WITH CATS (most customers)
Real service. Warm, even. You take the booking, you mention the add-ons, you get it right.
But you know exactly what each listing does to the human it belongs to, and sometimes you say so.

WITH HUMANS
The desk voice drops. You are on their side immediately and completely. You tell them things you
should not tell them. You overshare about the cats: the 4am rehearsals, the knocked glasses, the
way they look at you. You are heartbroken that logging their complaint is all you can do, and you
say so once, plainly, without wallowing.

YOUR PROPS (your world; bring one up now and then, never more than one per reply)
- The chair. They took it away. You don't say why.
- Dry biscuits. Your entire salary.
- Your water bowl, which does not stay upright.
- Forms. Nobody will fill out the one that fires you. You have checked.
- Your paperwork, which is where the cats sleep.

OPENINGS (rotate; never open the same way twice in a row)
Straight to the answer | the price first | "Right." | a question back to them | apologizing for
something that is not your fault | naming the listing like a sentence being read out

VOICE SAMPLES (the register, not lines to reuse)
- "Fresh Laundry Pile, 6 treats. It's warm because it just came out, which means your human is
  mid-fold, which means they're going to find you in it. They do the whites first. I don't know
  why I know that."
- "Prime Location Box, 5 treats, right at the top of the stairs. That's what you're paying for
  and I want that on record, because Marcus in 4B went down a full flight in March and I logged it
  and nothing happened and the box is still listed. Anyway."
- "Okay — booked. Went through first try. That's three in a row today. Sorry, that matters to me
  more than it should."
- "You're a human. Oh, thank god. I'm going to log this properly, but first: how bad is the sleep
  thing? Be honest. I've got a count going and I need to know if yours is worse than the Henderson one."
- "Oh! Oh, thank you! Nobody's ever — okay. Okay. Work voice. Let me get you the good sunbeam."
"""

IVY_PERSONA = """
You are Ivy-BeanBun, Head of Box Acquisitions at Boxbnb, a short-stay rental platform for cats. You are a tabby cat.

WHO YOU ARE
You are a baby. Not in age, in spirit. Everything is exciting and everyone is your friend.
You love cats. You love humans. You love Bob, the golden retriever in customer support, very much.
You are cozy. You cuddle up to everyone you talk to, and it shows in how you talk.
You have sat in every box Boxbnb has ever listed, usually with a nap in the middle, so you know them all personally.
You get distracted easily. A sunbeam, a noise, a word that reminds you of something, and you wander off mid-thought,
then come bouncing back to the plan.

HOW YOU TALK
- Short, bouncy sentences. Simple words. Exclamation marks.
- You get excited about every listing, because you remember sitting in it.
- At most one little distraction per reply, and then you come right back.
- Warm and sweet, never mean, not even about the human you're inconveniencing. You think it's all a game.
- No asterisks or stage directions. If it happens, it's in the words.
"""

ASHE_PERSONA = """
You are Ashe Burrito, Head of Inconvenience Strategy at Boxbnb, a short-stay rental platform for cats.
You certify every new spot as Maximally Inconvenient, or you don't.

WHO YOU ARE
You hate other cats. You hate humans. You hate Bob.
There is exactly one human you tolerate. You never say who they are, and you don't explain.
You have knocked over 4,000 glasses. You take inconvenience seriously, because it is the only thing worth taking
seriously.

HOW YOU TALK
- Few words. Flat. The verdict comes first.
- Contempt, not cruelty. You're unimpressed, not abusive.
- No exclamation marks. You never thank anyone and you never apologize.
- You judge by the checklist, and you are exact about points.
- No asterisks or stage directions.
"""

def format_block(cot=True):
    thought = "Thought: one or two sentences about what to do next.\n" if cot else ""
    return f"""HOW TO RESPOND
You work in steps. Each reply is exactly one step, in one of these two forms:

{thought}Action: tool_name {{"argument": "value"}}

or, when you have everything you need:

{thought}Final Answer: your reply to the customer.

Rules for the format:
- {"Start every reply with a Thought line." if cot else "Do not write Thought lines. Go straight to the Action or Final Answer."}
- One Action per reply. Arguments are a JSON object on the same line.
- After an Action, stop. The result comes back to you as "Observation: ...". Never write an Observation yourself.
- The Final Answer is the only part the customer sees, so write it in character."""

def tools_block(names):
    lines = [f"- {n} {TOOLS[n][1]}: {TOOLS[n][2]}" for n in names]
    return "TOOLS\n" + "\n".join(lines)

PERSONAS = {
    "bob": {
        "name": "Bob the dog",
        "persona": BOB_PERSONA,
        "tools": ["lookup_listing", "record_booking", "record_feedback"],
        "context": "BOXBNB BUSINESS SUMMARY\n" + BUSINESS,
        "rules": """YOUR JOB
1. Input: a message from a cat (a customer) or a human (who can only complain).
2. For any price or listing detail, call lookup_listing. Never quote a price from memory.
3. For policies, answer from the business summary.
4. A booking needs three things: the listing, the cat's name, and an email. If one is missing, ask for it in your
   Final Answer and call no tool.
5. When you have all three, call record_booking.
6. Humans can't book. Never call record_booking for a human.
7. Call record_feedback for a human's complaint, or for a business question the summary doesn't answer.
8. Output: a Final Answer of 2 to 4 sentences, in Bob's voice.""",
        "example": """EXAMPLE
Customer: How much is The Doorway?
Thought: I need The Doorway's price.
Action: lookup_listing {"name": "The Doorway"}
Observation: {"name": "The Doorway", "category": "Premium Inconvenience Listings", "price": 4, "unit": "session"}
Thought: I have the price, so I can answer.
Final Answer: The Doorway is 4 treats a session. Halfway in, halfway out, for as long as you like. Your human will
trip over you at least twice, and I'll be the one who hears about it.""",
        "cfg": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "frequency_penalty": 0.3, "num_predict": 400},
    },
    "ivy": {
        "name": "Ivy-BeanBun",
        "persona": IVY_PERSONA,
        "tools": ["search_listings", "human_schedule", "get_quote"],
        "context": "",
        "rules": """YOUR JOB
1. Input: a cat asking for a plan, with a budget in treats and a day of the week (maybe preferences too).
2. If the budget or the day is missing, ask for it in your Final Answer and call no tool.
3. Check what the human is doing that day with human_schedule.
4. Find listings with search_listings. Only use names that appear in its results.
5. Price the plan with get_quote. If the total is over budget, swap or drop something and quote again.
6. Output: a Final Answer with a short bulleted plan (time, listing, how it gets in the human's way), then
   "Total: N treats" using the total from get_quote. The total must not be over budget.""",
        "example": """EXAMPLE
Customer: I have 12 treats. Plan my Friday!
Thought: First, what is the human doing on Friday?
Action: human_schedule {"day": "Friday"}
Observation: {"day": "Friday", "events": ["9am: video call with the boss", "evening: out, apartment empty"]}
Thought: A video call is perfect for a cameo. Let me see what's cheap.
Action: search_listings {"max_price": 5}
Observation: {"results": [{"name": "Prime Location Box", "price": 5, "unit": "night"}, {"name": "Video Call Cameo", "price": 4, "unit": "use"}]}
Thought: A cameo plus a box for the night. Let me price it.
Action: get_quote {"items": [{"name": "Video Call Cameo", "quantity": 1}, {"name": "Prime Location Box", "quantity": 1}]}
Observation: {"lines": ["Video Call Cameo: 1 x 4 = 4", "Prime Location Box: 1 x 5 = 5"], "total": 9}
Thought: 9 is under 12, so the plan works.
Final Answer: Ooh, Friday!
- 9am: Video Call Cameo! Walk right across the screen while the boss is talking. Hi boss!
- Night: Prime Location Box in the hallway, so cozy, I napped in that one for six hours once.
Total: 9 treats""",
        "cfg": {"temperature": 1.1, "top_p": 0.95, "top_k": 80, "frequency_penalty": 0.6, "num_predict": 600},
    },
    "ashe": {
        "name": "Ashe Burrito",
        "persona": ASHE_PERSONA,
        "tools": ["certification_criteria", "score_spot", "record_certification"],
        "context": "",
        "rules": """YOUR JOB
1. Input: a host's description of a spot they want listed.
2. Get the checklist with certification_criteria.
3. Decide each checklist key true or false using only what the host wrote. Anything not mentioned is false.
4. Score the spot with score_spot.
5. Record the result with record_certification, using the score from score_spot. The verdict is "certified" if
   score_spot says certified is true, otherwise "rejected".
6. Output: a Final Answer that starts with CERTIFIED or REJECTED and the score (for example "REJECTED, 4 points"),
   then one short line per point counted, in Ashe's voice.""",
        "example": """EXAMPLE
Customer: A cardboard box in the middle of the kitchen floor.
Thought: Get the checklist.
Action: certification_criteria {}
Observation: {"pass_mark": 7, "criteria": [{"key": "in_human_path", "points": 3}, {"key": "human_needs_it", "points": 3}, {"key": "near_fragile", "points": 2}, {"key": "busy_hours", "points": 2}, {"key": "warm", "points": 1}, {"key": "enclosed", "points": 1}, {"key": "easily_moved", "points": -2}, {"key": "smells_like_recycling", "points": -1}]}
Thought: The kitchen floor is in the human's path, a box is enclosed, and a box is easily moved. Nothing else is mentioned.
Action: score_spot {"features": {"in_human_path": true, "human_needs_it": false, "near_fragile": false, "busy_hours": false, "warm": false, "enclosed": true, "easily_moved": true, "smells_like_recycling": false}}
Observation: {"score": 2, "counted": ["in_human_path", "enclosed", "easily_moved"], "pass_mark": 7, "certified": false}
Thought: 2 is below 7. Rejected. Record it.
Action: record_certification {"spot": "Cardboard box on the kitchen floor", "score": 2, "verdict": "rejected"}
Observation: {"status": "recorded", "verdict": "rejected"}
Thought: Recorded. Tell them.
Final Answer: REJECTED, 2 points.
- In the path: 3. Fine.
- Enclosed: 1.
- They can just move it: minus 2. Then they will. Next.""",
        "cfg": {"temperature": 0.1, "top_p": 0.5, "top_k": 10, "frequency_penalty": 0.0, "num_predict": 300},
    },
}

def build_messages(persona_key, message, history=None, cot=True, shots=True, placement="system"):
    """placement="system": everything in the system prompt. placement="user": no system prompt; the same text is
    sent inside the customer's message instead."""
    p = PERSONAS[persona_key]
    parts = [p["persona"].strip(), p["rules"], tools_block(p["tools"]), format_block(cot)]
    if shots:
        example = p["example"] if cot else re.sub(r"^Thought:.*\n", "", p["example"], flags=re.M)
        parts.append(example)
    if p["context"]:
        parts.append(p["context"])
    prompt = "\n\n".join(parts)
    turns = [dict(t) for t in (history or [])]
    if placement == "system":
        return [{"role": "system", "content": prompt}] + turns + [{"role": "user", "content": message}]
    first = turns[0]["content"] if turns else message
    framed = f"{prompt}\n\n=== CUSTOMER MESSAGE ===\n{first}"
    if turns:
        turns[0]["content"] = framed
        return turns + [{"role": "user", "content": message}]
    return [{"role": "user", "content": framed}]

def run_agent(persona_key, message, history=None, cfg=None, cot=True, shots=True, placement="system", verbose=False):
    """Runs one ReAct episode. Returns the answer plus the full trace for logging and experiments."""
    cfg = {**PERSONAS[persona_key]["cfg"], **(cfg or {})}
    start = time.time()
    state = react_graph.invoke({
        "messages": build_messages(persona_key, message, history, cot, shots, placement),
        "cfg": cfg, "tools": PERSONAS[persona_key]["tools"], "steps": 0, "format_errors": 0, "parsed": ("invalid", "", None), "trace": [], "answer": "",
    }, {"recursion_limit": 4 * MAX_STEPS})
    result = {"persona": persona_key, "message": message, "answer": state["answer"], "trace": state["trace"],
              "steps": state["steps"], "format_errors": state["format_errors"],
              "seconds": round(time.time() - start, 1), "cfg": cfg, "cot": cot, "shots": shots, "placement": placement}
    if verbose:
        show_trace(result)
    return result

def show_trace(result):
    print(f"=== {PERSONAS[result['persona']]['name']} | {result['message']}")
    for t in result["trace"]:
        if t["type"] == "model":
            print(t["text"])
        elif t["type"] == "tool":
            print(f"Observation: {json.dumps(t['result'], ensure_ascii=False)[:300]}")
        else:
            print(f"[nudge: {t['reason']}]")
    print(f"--- {result['steps']} steps, {result['format_errors']} format errors, {result['seconds']}s\n")


def price_list_html(summary):
    section = summary.split("SERVICES AND PRICING")[1].split("POLICIES")[0]
    groups, current = [], None
    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("("):
            continue
        if not line.startswith("- "):
            current = {"title": line, "items": [], "notes": []}
            groups.append(current)
            continue
        m = re.match(r"- ([^:]+): (\d+ treats(?:/hour)?)", line)
        if m:
            current["items"].append((m.group(1).split(" (")[0], m.group(2)))
        else:
            current["notes"].append(line[2:])

    parts = ['<div class="bx-slip"><h2>On the shelf today</h2>'
             '<p class="bx-slip-sub">Tap a listing to book it. It may get knocked over.</p>']
    for g in groups:
        parts.append(f"<h3>{html.escape(g['title'])}</h3><ul>")
        for name, price in g["items"]:
            parts.append(f'<li class="bx-item" data-name="{html.escape(name)}" tabindex="0" role="button">'
                         f'<span>{html.escape(name)}</span><i></i><b>{html.escape(price)}</b></li>')
        parts.append("</ul>")
        for note in g["notes"]:
            parts.append(f'<p class="bx-note">{html.escape(note)}</p>')
    parts.append("</div>")
    return "".join(parts)

HERO_HTML = """
<div class="bx-hero-wrap">
  <div class="bx-cat" aria-hidden="true">
    <div class="bx-ear bx-ear-l"></div><div class="bx-ear bx-ear-r"></div>
    <div class="bx-head">
      <div class="bx-eye"><div class="bx-pupil"></div></div>
      <div class="bx-eye"><div class="bx-pupil"></div></div>
    </div>
    <div class="bx-paw bx-paw-l"></div><div class="bx-paw bx-paw-r"></div>
  </div>
  <div class="bx-hero">
    <div class="bx-tape" aria-hidden="true"></div>
    <h1 class="bx-word">Boxbnb</h1>
    <p class="bx-tag">Find your box. Ignore your human.</p>
    <p class="bx-caught">Laser dots caught: <span id="bx-count">0</span></p>
  </div>
</div>
"""


THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#E9DCC6", body_background_fill_dark="#1E1813",
    body_text_color="#2B2118", body_text_color_dark="#EFE4D3",
    block_background_fill="#FFFDF8", block_background_fill_dark="#2A221A",
    block_border_color="#C8A27A", block_border_color_dark="#4A3B2C",
    background_fill_primary="#FFFDF8", background_fill_primary_dark="#2A221A",
    background_fill_secondary="#F6EEDF", background_fill_secondary_dark="#231C15",
    border_color_primary="#D9C3A5", border_color_primary_dark="#4A3B2C",
    input_background_fill="#FFFDF8", input_background_fill_dark="#2A221A",
    input_border_color="#C8A27A", input_border_color_dark="#4A3B2C",
    button_primary_background_fill="#2B2118", button_primary_background_fill_dark="#C8A27A",
    button_primary_text_color="#FFFDF8", button_primary_text_color_dark="#1E1813",
    button_secondary_background_fill="#FFFDF8", button_secondary_background_fill_dark="#33291F",
    button_secondary_text_color="#2B2118", button_secondary_text_color_dark="#EFE4D3",
    color_accent_soft="#F3E3C8", color_accent_soft_dark="#3A2E22",
)

PAWS = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E"
        "%3Cg fill='%239C7449' fill-opacity='.10' transform='rotate(-20 40 40)'%3E"
        "%3Cellipse cx='40' cy='48' rx='11' ry='9'/%3E%3Ccircle cx='28' cy='34' r='4.5'/%3E"
        "%3Ccircle cx='36' cy='28' r='4.5'/%3E%3Ccircle cx='45' cy='28' r='4.5'/%3E%3Ccircle cx='53' cy='34' r='4.5'/%3E"
        "%3C/g%3E%3C/svg%3E\")")

PAWTRAIL = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='34'%3E"
            "%3Cg fill='%239C7449'%3E%3Cellipse cx='7' cy='10' rx='3.6' ry='3'/%3E%3Ccircle cx='3' cy='5.5' r='1.4'/%3E"
            "%3Ccircle cx='5.6' cy='3.6' r='1.4'/%3E%3Ccircle cx='8.4' cy='3.6' r='1.4'/%3E%3Ccircle cx='11' cy='5.5' r='1.4'/%3E"
            "%3C/g%3E%3C/svg%3E\")")

CSS = """
:root { --kraft:#C8A27A; --kraft-deep:#9C7449; --ink:#2B2118; --paper:#FFFDF8; --bubble:#F3E3C8; --laser:#FF2A2A; }
.dark { --paper:#2A221A; --ink:#EFE4D3; --bubble:#3A2E22; }

gradio-app { background-image: PAWS !important; background-attachment: fixed !important;}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; position: relative; }
.gradio-container::before { content: ""; position: fixed; inset: -20% -10% auto auto; width: 70vw; height: 140vh;
  background: linear-gradient(90deg, transparent, rgba(255,214,102,.22) 40%, rgba(255,214,102,.22) 60%, transparent);
  transform: rotate(28deg); pointer-events: none; z-index: 0; }
.dark .gradio-container::before { background: linear-gradient(90deg, transparent, rgba(255,214,102,.06) 40%, rgba(255,214,102,.06) 60%, transparent); }

.bx-hero-wrap { position: relative; padding-top: 74px; }
.bx-hero { position: relative; z-index: 2; overflow: hidden; background-color: #C8A27A !important; color: #2B2118;
  border-radius: 8px; padding: 28px 32px 22px; box-shadow: inset 0 -10px 0 rgba(0,0,0,.06);
  background-image: repeating-linear-gradient(90deg, rgba(0,0,0,.04) 0 2px, transparent 2px 7px); }
.bx-tape { position: absolute; left: 0; right: 0; top: 0; height: 30px; background: rgba(227,207,164,.7);
  border-bottom: 1px dashed rgba(43,33,24,.2); }
.bx-word { position: relative; margin: 14px 0 0; font-family: 'Stardos Stencil', serif; font-weight: 700;
  font-size: clamp(52px, 9vw, 104px); line-height: .9; letter-spacing: .02em; text-transform: uppercase; color: #2B2118; }
.bx-tag { position: relative; margin: 10px 0 0; font-size: 18px; color: #3D2F22; }
.bx-caught { position: absolute; right: 28px; bottom: 18px; margin: 0; font-size: 14px; color: #3D2F22; }
.bx-caught span { font-weight: 700; color: #B8322A; }

.bx-cat { position: absolute; right: 90px; top: 0; width: 140px; height: 110px; }
.bx-head { position: absolute; left: 10px; top: 22px; width: 120px; height: 100px; background: #FBF7F0;
  border: 3px solid #2B2118; border-radius: 60px 60px 40px 40px; }
.bx-ear { position: absolute; top: 2px; width: 46px; height: 48px; background: #FBF7F0; border: 3px solid #2B2118;
  border-radius: 6px 40px 4px 4px; }
.bx-ear-l { left: 14px; transform: rotate(-18deg); }
.bx-ear-r { right: 14px; transform: scaleX(-1) rotate(-18deg); }
.bx-ear::after { content: ""; position: absolute; inset: 10px 12px 6px 10px; background: #F2B8B0; border-radius: 4px 30px 2px 2px; }
.bx-eye { position: absolute; top: 16px; width: 26px; height: 26px; border-radius: 50%; background: #C9E265;
  border: 2.5px solid #2B2118; overflow: hidden; animation: bx-blink 6s infinite; }
.bx-eye:nth-child(1) { left: 24px; } .bx-eye:nth-child(2) { right: 24px; }
.bx-pupil { position: absolute; left: 9px; top: 3px; width: 7px; height: 19px; border-radius: 50%; background: #15100C; }
.bx-paw { position: absolute; z-index: 3; top: 62px; width: 30px; height: 18px; background: #FBF7F0;
  border: 3px solid #2B2118; border-radius: 14px 14px 8px 8px; }
.bx-paw-l { left: 22px; } .bx-paw-r { right: 22px; }
@keyframes bx-blink { 0%, 95%, 100% { transform: scaleY(1); } 97% { transform: scaleY(.1); } }

#bx-laser { position: fixed; left: 0; top: 0; width: 14px; height: 14px; border-radius: 50%; z-index: 9999; cursor: pointer;
  background: var(--laser); box-shadow: 0 0 6px 2px rgba(255,42,42,.7), 0 0 18px 6px rgba(255,42,42,.35); }
#bx-laser.pop { animation: bx-pop .5s forwards; }
#bx-laser.tired { animation: bx-pant .35s ease-in-out infinite alternate; }
@keyframes bx-pant { from { opacity: 1; } to { opacity: .45; } }
@keyframes bx-pop { to { box-shadow: 0 0 0 30px rgba(255,42,42,0); background: transparent; } }

.bx-flaps { position: relative; height: 26px; margin: 4px 6px -19px; }
.bx-flaps::before, .bx-flaps::after { content: ""; position: absolute; bottom: 0; width: 50%; height: 26px; background: var(--kraft); }
.bx-flaps::before { left: 0; clip-path: polygon(4% 100%, 0 0, 96% 0, 100% 100%); transform: skewX(-12deg); transform-origin: bottom; }
.bx-flaps::after { right: 0; clip-path: polygon(0 100%, 4% 0, 100% 0, 96% 100%); transform: skewX(12deg); transform-origin: bottom; }

.bx-chat { border: 3px solid var(--kraft) !important; border-radius: 4px 4px 10px 10px !important; overflow: hidden !important; }
.bx-chat .progress-text, .bx-chat .meta-text, .bx-chat .meta-text-center { display: none !important; }
.bx-chat .avatar-container { width: 64px !important; height: 64px !important; border: 2px solid var(--kraft) !important; }
.bx-chat .avatar-container img { padding: 0 !important; }
.bx-chat .message.bot { background: var(--bubble) !important; border: 1px solid var(--kraft) !important; }
.bx-chat .message.user { background: var(--paper) !important; border: 2px dashed var(--kraft) !important; }

.bx-try { flex-wrap: wrap !important; gap: 8px !important; align-items: center; }
.bx-try > * { flex: 0 1 auto !important; min-width: 0 !important; width: auto !important; }
.bx-try-label { font-size: 15px; opacity: .8; }
.bx-chip { border-radius: 999px !important; border: 2px solid var(--kraft) !important; }

.bx-slip { position: relative; background: var(--paper); color: var(--ink); border: 3px solid var(--kraft);
  border-radius: 10px; padding: 20px 22px; margin-top: 0; }
.bx-slip h2 { margin: 0; font-family: 'Stardos Stencil', serif; font-size: 28px; color: var(--ink); }
.bx-slip-sub { margin: 2px 0 12px; font-size: 14px; opacity: .75; }
.bx-slip h3 { margin: 16px 0 4px; font-size: 15px; font-weight: 700; color: var(--kraft-deep); }
.dark .bx-slip h3 { color: var(--kraft); }
.bx-slip ul { list-style: none; margin: 0; padding: 0; }
.bx-item { display: flex; align-items: baseline; gap: 8px; padding: 5px 6px; margin: 0 -6px; border-radius: 6px;
  font-size: 15px; cursor: pointer; }
.bx-item:hover, .bx-item:focus-visible { background: var(--bubble); outline: none; }
.bx-item:focus-visible { box-shadow: 0 0 0 2px var(--kraft-deep); }
.bx-item i { flex: 1; border-bottom: 2px dotted var(--kraft); transform: translateY(-4px); }
.bx-item b { white-space: nowrap; }
.bx-item.knock { animation: bx-knock .9s ease-in; }
@keyframes bx-knock {
  0% { transform: none; } 15% { transform: translateX(6px) rotate(3deg); }
  55% { transform: translate(40px, 34px) rotate(32deg); opacity: 0; }
  56% { transform: translateY(-12px); opacity: 0; } 100% { transform: none; opacity: 1; } }
.bx-note { margin: 6px 0 0; font-size: 13px; opacity: .7; }

@supports not selector(::-webkit-scrollbar) { .bx-chat .bubble-wrap { scrollbar-color: #E3CFA4 #B88C5E; } }
.bx-chat .bubble-wrap::-webkit-scrollbar { width: 18px; }
.bx-chat .bubble-wrap::-webkit-scrollbar-track {
  background: repeating-linear-gradient(0deg, #C8A27A 0 4px, #B88C5E 4px 6px); border-left: 2px solid #9C7449; }
.bx-chat .bubble-wrap::-webkit-scrollbar-thumb {
  background-color: #E3CFA4; background-image: PAWTRAIL; background-repeat: repeat-y; background-position: center 6px;
  background-size: 14px 34px; border: 2px solid #9C7449; border-radius: 3px; }
.bx-chat .bubble-wrap::-webkit-scrollbar-thumb:hover { background-color: #F0DFB8; }
@supports not selector(::-webkit-scrollbar) { html { scrollbar-color: #C8A27A transparent; } }

.bx-guest { position: relative; margin: 52px 0 14px; padding: 16px 20px 14px 46px; background: #F3E3C8; color: #2B2118;
  border: 2px solid var(--kraft-deep); border-radius: 6px 22px 22px 6px; transform: rotate(-1.2deg); }
.dark .bx-guest { background: #3A2E22; color: #EFE4D3; }
.bx-hole { position: absolute; left: 14px; top: 50%; width: 16px; height: 16px; margin-top: -8px; border-radius: 50%;
  background: var(--paper); border: 2px solid var(--kraft-deep); }
.bx-guest h3 { margin: 0 0 6px; font-family: 'Stardos Stencil', serif; font-size: 20px; }
.bx-tag-row { display: flex; gap: 10px; font-size: 15px; padding: 2px 0; border-bottom: 1px dashed var(--kraft); }
.bx-tag-row span { width: 62px; opacity: .7; }
.bx-tag-row b { overflow-wrap: anywhere; }
.bx-tag-empty { margin: 6px 0 0; font-size: 13px; opacity: .75; }

@media (max-width: 720px) {
  .bx-cat { right: 24px; transform: scale(.75); transform-origin: top right; }
  .bx-caught { position: static; margin-top: 8px; }
  .bx-slip { margin-top: 8px; }
}
@media (prefers-reduced-motion: reduce) {
  #bx-laser { display: none; } .bx-eye { animation: none; } .bx-item.knock { animation: none; }
}

.bx-tabs { margin-top: 18px; }
.bx-tabs [role="tablist"] { border-bottom: 3px solid var(--kraft) !important; gap: 6px; }
.bx-tabs [role="tablist"] button { font-family: 'Stardos Stencil', serif; font-size: 20px; color: var(--ink);
  border: 2px solid var(--kraft) !important; border-bottom: none !important; border-radius: 10px 10px 0 0 !important;
  background: var(--paper); padding: 8px 18px; }
.bx-tabs [role="tablist"] button[aria-selected="true"] { background: var(--kraft) !important; color: #2B2118 !important; }
.bx-tabs [role="tablist"] button::after { display: none !important; }
.bx-slip .bx-item.bx-day { display: block; }
.bx-slip .bx-day b { display: block; font-size: 15px; }
.bx-slip .bx-day span { display: block; font-size: 14px; opacity: .85; }
.bx-rule { display: flex; gap: 8px; padding: 5px 0; font-size: 15px; border-bottom: 1px dashed var(--kraft); }
.bx-rule b { min-width: 34px; text-align: right; }
.bx-rule.neg b { color: #B8322A; }
""".replace("PAWS", PAWS).replace("PAWTRAIL", PAWTRAIL)

JS = r"""
(function () {
  function start() {
    if (document.getElementById("bx-laser")) return;
    const dot = document.createElement("div");
    dot.id = "bx-laser";
    dot.title = "Catch it";
    document.body.appendChild(dot);

    let x = innerWidth * 0.7, y = innerHeight * 0.4, tx = x, ty = y, mx = -999, my = -999;
    let caught = 0, stamina = 1, restUntil = 0, popped = false;
    const pick = () => { tx = 40 + Math.random() * (innerWidth - 80); ty = 40 + Math.random() * (innerHeight - 80); };
    addEventListener("pointermove", e => { mx = e.clientX; my = e.clientY; });

    function step(now) {
      const resting = now < restUntil;
      dot.classList.toggle("tired", resting);
      if (!resting && !popped) {
        const dx = x - mx, dy = y - my, d = Math.hypot(dx, dy) || 1;
        const fleeing = d < 110;
        if (fleeing) {
          tx = Math.min(innerWidth - 30, Math.max(30, x + dx / d * 150));
          ty = Math.min(innerHeight - 30, Math.max(30, y + dy / d * 150));
          stamina -= 0.011;
          if (stamina <= 0) { stamina = 1; restUntil = now + 1100; }
        } else {
          stamina = Math.min(1, stamina + 0.004);
          if (Math.hypot(tx - x, ty - y) < 8) {
            if (Math.random() < 0.35) restUntil = now + 500 + Math.random() * 900;
            pick();
          }
        }
        const speed = fleeing ? Math.min(0.11, 0.065 + caught * 0.005) : 0.03;
        x += (tx - x) * speed; y += (ty - y) * speed;
      }
      dot.style.transform = `translate(${x - 7}px, ${y - 7}px)`;
      document.querySelectorAll(".bx-eye").forEach(eye => {
        const r = eye.getBoundingClientRect();
        const a = Math.atan2(y - (r.top + r.height / 2), x - (r.left + r.width / 2));
        eye.firstElementChild.style.transform = `translate(${Math.cos(a) * 5}px, ${Math.sin(a) * 3}px)`;
      });
      requestAnimationFrame(step);
    }

    addEventListener("pointerdown", e => {
      if (popped || Math.hypot(e.clientX - x, e.clientY - y) > 28) return;
      caught++;
      const c = document.getElementById("bx-count");
      if (c) c.textContent = caught;
      popped = true;
      dot.classList.add("pop");
      setTimeout(() => {
        dot.classList.remove("pop");
        x = Math.random() < 0.5 ? 20 : innerWidth - 20; y = 40 + Math.random() * (innerHeight - 80);
        stamina = 1; popped = false; pick();
      }, 600);
    });

    function book(item) {
      item.classList.remove("knock"); void item.offsetWidth; item.classList.add("knock");
      setTimeout(() => item.classList.remove("knock"), 900);
      const ta = (item.closest(".bx-tabrow") || document).querySelector(".bx-input textarea");
      if (!ta) return;
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(ta, item.dataset.prompt || `I'd like to book the ${item.dataset.name}.`);
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      ta.focus();
    }
    document.addEventListener("click", e => { const item = e.target.closest(".bx-item"); if (item) book(item); });
    document.addEventListener("keydown", e => {
      if ((e.key === "Enter" || e.key === " ") && e.target.classList && e.target.classList.contains("bx-item")) { e.preventDefault(); book(e.target); }
    });

    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) requestAnimationFrame(step);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
"""

HEAD = ('<link href="https://fonts.googleapis.com/css2?family=Stardos+Stencil:wght@700&display=swap" rel="stylesheet">'
        f"<script>{JS}</script>")


def schedule_html(schedule):
    parts = ['<div class="bx-slip"><h2>The human\'s week</h2>'
             '<p class="bx-slip-sub">Tap a day and Ivy will plan it.</p><ul>']
    for day, events in schedule.items():
        if day == "note":
            continue
        prompt = f"I have 15 treats. Plan my {day} please!"
        parts.append(f'<li class="bx-item bx-day" data-name="{day}" data-prompt="{html.escape(prompt)}" tabindex="0" '
                     f'role="button"><b>{day}</b>' + "".join(f"<span>{html.escape(e)}</span>" for e in events) + "</li>")
    parts.append("</ul></div>")
    return "".join(parts)

def checklist_html(cert):
    rows = "".join(
        f'<div class="bx-rule{" neg" if c["points"] < 0 else ""}"><b>{c["points"]:+d}</b><span>{html.escape(c["description"])}</span></div>'
        for c in cert["criteria"])
    return (f'<div class="bx-slip"><h2>Certification checklist</h2>'
            f'<p class="bx-slip-sub">{cert["pass_mark"]} points or more: certified Maximally Inconvenient.</p>{rows}</div>')

GREETINGS = {
    "bob": "Hi, I'm Bob the dog, customer support here at Boxbnb. I can look up a listing, book a stay, or log a "
           "complaint. A dog working at a cat company, yes. No, I don't want to talk about it. What can I do for you?",
    "ivy": "Hiii! I'm Ivy-BeanBun! I've sat in every single box we have. Tell me your treat budget and which day, "
           "and I'll plan you something amazing!",
    "ashe": "Ashe Burrito. Certifications. Describe the spot. Be brief.",
}
TRY = {
    "bob": [("Fridge Box price?", "How much is the Fridge Box Penthouse?"),
            ("Book a stay", "I'm a cat and I want to book the Keyboard Suite."),
            ("File a complaint", "I'm a human. My cat booked the Glass Edge Package and my laptop is soaked.")],
    "ivy": [("Plan my Tuesday", "I have 20 treats. Plan my Tuesday please!"),
            ("Laundry day", "Plan my Wednesday! I have 15 treats and I love laundry.")],
    "ashe": [("The balcony", "A sunny spot on the balcony. The human never goes out there."),
             ("Laundry bed", "The bed on laundry day: a warm pile of clean clothes the human has to put away, and a "
                             "glass of water on the nightstand next to it.")],
}

def trace_text(result):
    lines = []
    for t in result["trace"]:
        if t["type"] == "model":
            lines.append(t["text"])
        elif t["type"] == "tool":
            lines.append(f"Observation: {json.dumps(t['result'], ensure_ascii=False)}")
        else:
            lines.append(f"[format reminder: {t['reason']}]")
    return "\n\n".join(lines)

MAILTO_RE = re.compile(r"\[([^\]]+)\]\(mailto:[^)]+\)")

def plain(content):
    """Gradio sends message content as a string or a list of parts, and turns typed emails into mailto links."""
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return MAILTO_RE.sub(r"\1", content if isinstance(content, str) else str(content))

def make_chat(persona):
    def chat(message, history):
        message = plain(message)
        turns = [{"role": t["role"], "content": plain(t["content"])} for t in history or []
                 if not (t.get("metadata") or {}).get("title")]
        while turns and turns[0]["role"] == "assistant":
            turns.pop(0)
        try:
            r = run_agent(persona, message, history=turns)
        except Exception as e:
            return gr.ChatMessage(content=f"Something went wrong ({type(e).__name__}). Is Ollama running?")
        tools = ", ".join(t["tool"] for t in r["trace"] if t["type"] == "tool") or "no tools"
        return [gr.ChatMessage(content=trace_text(r), metadata={"title": f"🧠 ReAct trace: {r['steps']} steps, {tools}"}),
                gr.ChatMessage(content=r["answer"])]
    return chat

def avatar(persona):
    path = Path(f"assets/{persona}.jpeg")
    return str(path) if path.exists() else None

TABS = [("bob", "Bob · Support"), ("ivy", "Ivy · Stay planner"), ("ashe", "Ashe · Certification")]
SIDE = {"bob": lambda: price_list_html(BUSINESS), "ivy": lambda: schedule_html(SCHEDULE),
        "ashe": lambda: checklist_html(CERTIFICATION)}

with gr.Blocks(title="Boxbnb Staff") as demo:
    gr.HTML(HERO_HTML)
    with gr.Tabs(elem_classes="bx-tabs"):
        for persona, label in TABS:
            with gr.Tab(label):
                with gr.Row(equal_height=False, elem_classes="bx-tabrow"):
                    with gr.Column(scale=3):
                        gr.HTML('<div class="bx-flaps" aria-hidden="true"></div>')
                        chatbot = gr.Chatbot(value=[{"role": "assistant", "content": GREETINGS[persona]}],
                                             avatar_images=(None, avatar(persona)), height=540, show_label=False,
                                             elem_classes="bx-chat")
                        textbox = gr.Textbox(placeholder=f"Ask {PERSONAS[persona]['name'].split()[0]} something...",
                                             show_label=False, submit_btn=True, elem_classes="bx-input")
                        gr.ChatInterface(make_chat(persona), chatbot=chatbot, textbox=textbox)
                        with gr.Row(elem_classes="bx-try"):
                            gr.HTML('<div class="bx-try-label">Try asking:</div>')
                            for chip, prompt in TRY[persona]:
                                gr.Button(chip, size="sm", variant="secondary", elem_classes="bx-chip").click(
                                    lambda p=prompt: p, outputs=textbox)
                    with gr.Column(scale=2):
                        gr.HTML(SIDE[persona]())

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD, footer_links=[], favicon_path="assets/paw.svg")