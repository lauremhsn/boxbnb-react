"""Which sampling settings does the local Ollama model actually honor? Needs `ollama serve` running."""
import ollama

MODEL = "llama3.1:8b"
NAME = "Invent a name for a cat cafe. Reply with the name only."
REPEAT = "Write the word box 40 times, separated by spaces. Nothing else."

def ask(prompt, **options):
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options=options)
    return r["message"]["content"].strip()

print("temperature 0:  ", [ask(NAME, temperature=0) for _ in range(3)])
print("temperature 1.5:", [ask(NAME, temperature=1.5) for _ in range(3)])
print("top_k=1 @ 1.5:  ", [ask(NAME, temperature=1.5, top_k=1) for _ in range(3)])

base = dict(temperature=0.8, seed=42)
baseline = ask(REPEAT, **base)
print(f"\nbaseline (seed 42): {baseline.count('box')} x 'box'")
for name, extra in [("top_p=0.1", {"top_p": 0.1}),
                    ("repeat_penalty=1.8", {"repeat_penalty": 1.8}),
                    ("presence_penalty=1.5", {"presence_penalty": 1.5}),
                    ("frequency_penalty=1.5", {"frequency_penalty": 1.5}),
                    ("num_predict=10", {"num_predict": 10})]:
    out = ask(REPEAT, **base, **extra)
    verdict = "same as baseline, probably IGNORED" if out == baseline else "changed, honored"
    print(f"{name:<22} {out.count('box'):>3} x 'box' | {verdict} | {out[:60]!r}")
