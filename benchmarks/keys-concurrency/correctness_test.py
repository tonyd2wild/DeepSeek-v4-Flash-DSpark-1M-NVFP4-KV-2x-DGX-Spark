#!/usr/bin/env python3
"""Decisive Patch-1 correctness test under continuous-batch CONDENSE.

A deterministic 'victim' request (temp 0, ignore_eos) is run:
  (1) ALONE -> reference output
  (2) WHILE 'churner' requests start and finish at staggered times (forcing
      the running batch to condense, which moves the victim's batch row).
If main_kv_cache is correctly keyed by request slot (Patch 1), the victim's
output is IDENTICAL in both cases. Pre-patch, condense corrupts it -> divergence.
"""
import json, time, urllib.request, sys, re, threading

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.100.10.2:8888"
MODEL = "deepseek-v4-flash-dspark"
VICTIM = ("Write a long, deterministic enumerated list of the first 60 prime "
          "numbers, one per line as 'n: prime', with no commentary.")
CHURN = ("Say a short greeting.")

def chat(prompt, max_tokens, ignore_eos=True):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],
          "temperature":0.0,"max_tokens":max_tokens,"stream":False,
          "ignore_eos":ignore_eos,"chat_template_kwargs":{"thinking":False}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=600) as r:
        o=json.loads(r.read().decode())
    return o["choices"][0]["message"]["content"]

def metrics_acc():
    try:
        with urllib.request.urlopen(BASE+"/metrics",timeout=10) as r: t=r.read().decode()
    except: return (0,0)
    g=lambda n: sum(float(x) for x in re.findall(r"^%s(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"%re.escape(n),t,re.M)) or 0.0
    return (g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_draft_tokens_total"))

print("warmup..."); chat(CHURN,8)
print("=== (1) victim ALONE -> reference ===")
ref = chat(VICTIM, 220)
print(f"reference length: {len(ref)} chars")

print("=== (2) victim WHILE churners start/finish (condense stress) ===")
stop=False
def churn_loop():
    i=0
    while not stop:
        try: chat(CHURN, 16)   # short -> finishes fast -> condense events
        except: pass
        i+=1
    print(f"  churn requests fired: {i}")
a0,d0=metrics_acc()
threads=[threading.Thread(target=churn_loop) for _ in range(6)]
for t in threads: t.start()
time.sleep(1)  # let churn batch fill
victim2 = chat(VICTIM, 220)
stop=True
for t in threads: t.join()
a1,d1=metrics_acc()

print(f"\nvictim-under-churn length: {len(victim2)} chars")
identical = (ref == victim2)
print(f"OUTPUT IDENTICAL to reference: {identical}")
if not identical:
    # show first divergence
    for i,(x,y) in enumerate(zip(ref,victim2)):
        if x!=y:
            print(f"  first diff at char {i}: ref={ref[i:i+40]!r} got={victim2[i:i+40]!r}"); break
    print(f"  ref tail:  ...{ref[-120:]!r}")
    print(f"  got tail:  ...{victim2[-120:]!r}")
acc = (a1-a0)/(d1-d0) if d1>d0 else float('nan')
print(f"acceptance under churn: {acc:.3f}  (healthy ~0.6, collapsed if ~0)")
print("\nVERDICT:", "PASS — condense-safe (Patch 1 works)" if identical and acc>0.4 else "FAIL — corruption under condense")
