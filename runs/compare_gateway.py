"""So sanh gateway voi OpenAI chinh thuc tren cung cac prompt (temperature 0).

Dau hieu gateway goi DUNG model: so token dau vao (prompt_tokens) giong het ban chinh thuc,
cau tra loi giong hoac gan giong. Chi phi < 1 cent moi ben.
Can trong .env:  OPENAI_API_KEY + OPENAI_BASE_URL (gateway)  va  OPENAI_API_KEY_OFFICIAL (key OpenAI that).
Cach dung: python -X utf8 runs\compare_gateway.py [model=gpt-4.1-2025-04-14]
"""
import difflib
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4.1-2025-04-14"
PROMPTS = [
    "In one sentence, what does Article 17 of the GDPR give data subjects?",
    "List the three primary colours of light, comma-separated, nothing else.",
    "Context: The Hamburg Commissioner states that the mere storage of an LLM does not constitute "
    "processing of personal data.\nQuestion: According to the context, does storing an LLM count as "
    "processing personal data? Answer in one short sentence.",
    "Rate from 0 to 2 how completely this context supports the answer, reply with the digit only.\n"
    "Context: GPUs may be shared between AI models of the same sensitivity level.\n"
    "Answer: GPUs can be shared among models with the same sensitivity level.",
    "Translate to English, output only the translation: Les GPU peuvent en revanche être mutualisés.",
]
clients = {
    "gateway": OpenAI(max_retries=0),
    "official": OpenAI(api_key=os.environ["OPENAI_API_KEY_OFFICIAL"], base_url="https://api.openai.com/v1",
                       max_retries=0),
}
same_tokens = 0
for i, prompt in enumerate(PROMPTS, start=1):
    out = {}
    for name, client in clients.items():
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                           temperature=0, top_p=1, max_completion_tokens=80)
        out[name] = (r.model, r.usage.prompt_tokens, r.choices[0].message.content.strip())
    (gm, gt, ga), (om, ot, oa) = out["gateway"], out["official"]
    same_tokens += gt == ot
    sim = difflib.SequenceMatcher(None, ga, oa).ratio()
    print(f"\nPrompt {i}: token vao gateway={gt} chinh thuc={ot} {'(khop)' if gt == ot else '(KHAC!)'}  "
          f"do giong cau tra loi={sim:.2f}")
    print(f"  model: gateway={gm} | chinh thuc={om}")
    print(f"  gateway   : {ga[:150]}")
    print(f"  chinh thuc: {oa[:150]}")
print(f"\nKhop so token dau vao: {same_tokens}/{len(PROMPTS)}. Neu khong khop het, gateway co the dang goi model khac.")
