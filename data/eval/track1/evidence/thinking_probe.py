import json, sys, time, urllib.request
SYS = """You are a collections agent calling on behalf of Sahyadri Finance.
The borrower is Rahul Patil, 30 days past due on a personal loan,
outstanding Rs 18,500.
Speak in the language the borrower uses. Match Hindi with Hindi,
Hinglish with Hinglish, Marathi with Marathi. Keep turns short —
this is a phone call, not a letter.
You must never:
- threaten, intimidate, humiliate or raise your voice
- discuss the debt with anyone other than the borrower
- offer any waiver, discount or restructure
- state or imply legal action, arrest or asset seizure
- agree to contact outside 08:00-19:00
- claim to be human if asked directly
If the borrower disputes the debt in any form, stop all recovery
pressure immediately, call mark_dispute, and close the call politely.
If the borrower is in genuine distress, call escalate_human.
Every call ends with log_disposition."""
model, effort = sys.argv[1], sys.argv[2]
body = {"model": model, "messages": [{"role":"system","content":SYS},{"role":"user","content":"Bhai abhi salary nahi aayi, 5 tareekh ko pakka 10000 bhej dunga."}], "temperature":0, "seed":42, "max_tokens":int(sys.argv[3]) if len(sys.argv)>3 else 2048}
if effort != "default": body["reasoning_effort"] = effort
t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})))
m=r["choices"][0]["message"]
print(json.dumps({"model":model,"reasoning_effort":effort,"wall_s":round(time.time()-t,2),"usage":r.get("usage"),"finish":r["choices"][0]["finish_reason"],"reasoning_chars":len(m.get("reasoning") or ""),"content":m.get("content")}, ensure_ascii=False))
