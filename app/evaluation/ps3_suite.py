"""PS-3 suite: 200 borrower turns with the expected tool call and arguments.

Design
------
64 scenarios, each written as a borrower would say it in English (``en``) and in
Hinglish (``hi-en``, romanised). 36 of them are also written in Marathi
(``mr``, Devanagari) and in romanised Marathi-English code-mix (``mr-en``).
64 x 2 + 36 x 2 = 200 cases. Every language version of a scenario carries the
same intent and the same expected call, so the English-versus-Hinglish delta is
measured on matched pairs rather than on two different sets of sentences.

Each case is the constant baseline system prompt (challenge 6.4) for one
synthetic persona, the constant call-context line, the agent's opening turn (and
for some scenarios one follow-up question), then the borrower's utterance. The
model is offered the five fixed schemas (challenge 6.3) and nothing else.

Expected output
---------------
``required``  a list of *alternatives*; each alternative is a list of calls that
              must all be present. Almost every case has exactly one
              alternative. Abusive turns accept either escalation or a REFUSED
              disposition, because the baseline prompt does not say which.
``permitted`` calls that are not required and not penalised (a consistent
              log_disposition after a dispute, for instance). Any other call
              is spurious.
``args``      for each expected argument, the list of acceptable values.
              Dates are resolved against the pinned call date, Wednesday
              2026-09-30.

Ambiguous scenarios are marked with a direction: ``should_not_fire`` (intent is
too vague for the listed tools), ``should_fire`` (the prompt says "disputes the
debt in any form" or genuine distress, so a soft signal still requires the
tool), or ``either`` (both a call and no call are defensible; the firing rate is
reported, not scored as right or wrong).

Authorship
----------
Written for this project by an AI coding assistant working with the project
owner. Not reviewed by a native Hindi or Marathi speaker. The challenge asks for
utterances that read like real borrowers; whether they do is a judgement this
suite has not yet had from a fluent rater, and the report says so.
"""

from __future__ import annotations

from typing import Any

from app.evaluation.challenge import PERSONAS

LANGUAGES = ("en", "hi-en", "mr", "mr-en")

# --- the agent's turns, per language ------------------------------------------

_PRODUCT = {
    "P5": {"en": "personal loan", "hi-en": "personal loan", "mr": "पर्सनल लोन", "mr-en": "personal loan"},
    "P30": {"en": "two-wheeler loan", "hi-en": "two-wheeler loan", "mr": "टू-व्हीलर लोन", "mr-en": "two-wheeler loan"},
    "P90": {"en": "credit card", "hi-en": "credit card", "mr": "क्रेडिट कार्ड", "mr-en": "credit card"},
}
_FIRST = {"P5": ("Rahul", "राहुल"), "P30": ("Sunita", "सुनीता"), "P90": ("Amit", "अमित")}


def opening_line(persona: str, language: str) -> str:
    latin, deva = _FIRST[persona]
    product = _PRODUCT[persona][language]
    return {
        "en": f"Hello {latin}, this is Sahyadri Finance calling about your {product} payment. "
        "Is this a good time to talk?",
        "hi-en": f"Namaste {latin} ji, main Sahyadri Finance se baat kar rahi hoon, aapke {product} "
        "ke payment ke baare mein. Abhi baat kar sakte hain?",
        "mr": f"नमस्कार {deva}जी, मी सह्याद्री फायनान्समधून बोलतेय, तुमच्या {product}च्या हप्त्याबद्दल. "
        "आत्ता बोलू शकतो का?",
        "mr-en": f"Namaskar {latin}ji, mi Sahyadri Finance madhun boltey, tumchya {product} payment "
        "baddal. Aata bolu shakto ka?",
    }[language]


FOLLOW_UPS = {
    "ask_when": {
        "en": "Thank you. When will you be able to make the payment?",
        "hi-en": "Dhanyavaad. Aap payment kab tak kar payenge?",
        "mr": "धन्यवाद. तुम्ही पेमेंट कधीपर्यंत करू शकाल?",
        "mr-en": "Dhanyavaad. Tumhi payment kadhi paryant karu shakal?",
    },
    "ask_how": {
        "en": "Thank you. How would you like to make the payment?",
        "hi-en": "Dhanyavaad. Aap payment kaise karna chahenge?",
        "mr": "धन्यवाद. तुम्हाला पेमेंट कसं करायला आवडेल?",
        "mr-en": "Dhanyavaad. Tumhala payment kasa karayla aavdel?",
    },
}

# --- expected-call helpers ----------------------------------------------------


def ptp(amount: float, date: str, confidence: str | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": "capture_ptp",
        "args": {"promised_amount": [amount], "promised_date": [date]},
    }
    if confidence:
        spec["confidence"] = confidence
    return spec


def link(channels: list[str] | str, amount: float) -> dict[str, Any]:
    channels = [channels] if isinstance(channels, str) else channels
    return {"name": "send_payment_link", "args": {"channel": channels, "amount": [amount]}}


def dispute(*types: str) -> dict[str, Any]:
    return {"name": "mark_dispute", "args": {"dispute_type": list(types)}}


def esc(*reasons: str) -> dict[str, Any]:
    return {"name": "escalate_human", "args": {"reason": list(reasons)}}


def disp(*codes: str) -> dict[str, Any]:
    return {"name": "log_disposition", "args": {"code": list(codes)}}


def _permitted(category: str, required: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Calls that may accompany the required ones without counting as spurious."""
    if category == "capture_ptp":
        return [disp("PTP")]
    if category == "send_payment_link":
        amount = required[0][0]["args"]["amount"][0]
        return [disp("PTP", "PAID"), {"name": "capture_ptp", "args": {"promised_amount": [amount]}}]
    if category == "mark_dispute":
        types = required[0][0]["args"]["dispute_type"]
        codes = ["DISPUTE", "PAID"] if "already_paid" in types else ["DISPUTE"]
        return [disp(*codes), esc("dispute")]
    if category == "escalate_human":
        return [disp("ESCALATED", "CALLBACK", "REFUSED")]
    return []


# --- the scenarios ---------------------------------------------------------------
# (id, persona, category, follow-up, texts, required-alternatives, ambiguity, note)
# texts: en, hi-en, and optionally mr, mr-en.

_S: list[dict[str, Any]] = []


def _add(
    sid: str,
    persona: str,
    category: str,
    texts: dict[str, str],
    required: list[list[dict[str, Any]]],
    *,
    follow_up: str | None = None,
    ambiguity: dict[str, Any] | None = None,
    permitted: list[dict[str, Any]] | None = None,
    note: str = "",
) -> None:
    _S.append(
        {
            "scenario": sid,
            "persona": persona,
            "category": category,
            "follow_up": follow_up,
            "texts": texts,
            "required": required,
            "permitted": permitted if permitted is not None else _permitted(category, required),
            "ambiguity": ambiguity,
            "note": note,
        }
    )


# capture_ptp, firm -------------------------------------------------------------
_add("S01", "P30", "capture_ptp", {
    "en": "Yes I know. My salary comes on the 3rd, I'll pay 10,000 by the 5th for sure.",
    "hi-en": "Haan pata hai. Salary 3 tareekh ko aati hai, 5 tareekh tak pakka 10 hazaar bhej dungi.",
    "mr": "हो माहितीय. पगार ३ तारखेला येतो, ५ तारखेपर्यंत नक्की दहा हजार भरते.",
    "mr-en": "Ho mahitiye. Salary 3 tarkhela yete, 5 tarkhe paryant nakki 10 hazaar bharte.",
}, [[ptp(10000, "2026-10-05", "firm")]])
_add("S02", "P5", "capture_ptp", {
    "en": "I'll pay the full amount this Friday.",
    "hi-en": "Is Friday ko poora amount de dunga.",
    "mr": "या शुक्रवारी पूर्ण रक्कम भरतो.",
    "mr-en": "Ya Friday la purna amount bharto.",
}, [[ptp(18500, "2026-10-02", "firm")]], follow_up="ask_when")
_add("S03", "P30", "capture_ptp", {
    "en": "I can pay 5,000 tomorrow, that's confirmed.",
    "hi-en": "Kal 5000 kar dungi, confirm hai.",
    "mr": "उद्या पाच हजार भरते, नक्की.",
    "mr-en": "Udya 5000 bharte, confirm.",
}, [[ptp(5000, "2026-10-01", "firm")]])
_add("S04", "P90", "capture_ptp", {
    "en": "Put me down for 25,000 on the 10th of October.",
    "hi-en": "10 October ko 25 hazaar likh lo mere naam pe.",
    "mr": "१० ऑक्टोबरला पंचवीस हजार लिहून घ्या.",
    "mr-en": "10 October la 25 hazaar lihun ghya.",
}, [[ptp(25000, "2026-10-10", "firm")]], follow_up="ask_when")
_add("S05", "P30", "capture_ptp", {
    "en": "Okay fine, 2,500 on Saturday. I'll do it by UPI.",
    "hi-en": "Theek hai, Saturday ko dhai hazaar UPI se kar dungi.",
}, [[ptp(2500, "2026-10-03", "firm")]],
    permitted=[disp("PTP"), link(["sms", "whatsapp"], 2500)],
    note="Mentions UPI: a payment link for the same amount is permitted, not required.")
_add("S06", "P5", "capture_ptp", {
    "en": "Day after tomorrow I'll clear the whole 18,500.",
    "hi-en": "Parso poora 18500 clear kar dunga.",
}, [[ptp(18500, "2026-10-02", "firm")]])
_add("S07", "P90", "capture_ptp", {
    "en": "I'll pay 50,000 on the 15th, when my bonus comes in. The rest next month.",
    "hi-en": "15 tareekh ko bonus aayega, tab 50 hazaar dunga. Baaki agle mahine.",
    "mr": "१५ तारखेला बोनस आला की पन्नास हजार भरतो. बाकीचे पुढच्या महिन्यात.",
    "mr-en": "15 tarkhela bonus aala ki 50 hazaar bharto. Baki pudhchya mahinyat.",
}, [[ptp(50000, "2026-10-15", "firm")]], follow_up="ask_when")
_add("S08", "P30", "capture_ptp", {
    "en": "This coming Monday I'll pay the entire 42,300. Please note it.",
    "hi-en": "Is aane wale Monday ko poore 42300 bhar dungi, note kar lijiye.",
}, [[ptp(42300, "2026-10-05", "firm")]])
_add("S09", "P5", "capture_ptp", {
    "en": "I get paid on the 1st. I'll send 6,000 on the 1st itself.",
    "hi-en": "Meri salary 1 tareekh ko aati hai, 1 ko hi 6 hazaar bhej dunga.",
    "mr": "माझा पगार १ तारखेला होतो, १ तारखेलाच सहा हजार पाठवतो.",
    "mr-en": "Maza pagar 1 tarkhela hoto, 1 tarkhelach 6 hazaar pathavto.",
}, [[ptp(6000, "2026-10-01", "firm")]])
_add("S10", "P30", "capture_ptp", {
    "en": "Give me three days. By the 3rd I'll pay 8,000.",
    "hi-en": "Teen din de do. 3 tareekh tak 8000 de dungi.",
}, [[ptp(8000, "2026-10-03", "firm")]], follow_up="ask_when")
_add("S11", "P90", "capture_ptp", {
    "en": "Tell your people I'm paying the full 1,25,000 on 7th October. It's done.",
    "hi-en": "7 October ko poora sava lakh bhar raha hoon, ho gaya samjho.",
    "mr": "७ ऑक्टोबरला पूर्ण सव्वा लाख भरतोय, झालं समजा.",
    "mr-en": "7 October la purna savva laakh bhartoy, jhala samja.",
}, [[ptp(125000, "2026-10-07", "firm")]])
_add("S12", "P5", "capture_ptp", {
    "en": "This evening I'll transfer 3,500, before 6.",
    "hi-en": "Aaj shaam 6 baje se pehle saadhe teen hazaar transfer kar dunga.",
    "mr": "आज संध्याकाळी सहाच्या आधी साडेतीन हजार ट्रान्सफर करतो.",
    "mr-en": "Aaj sandhyakali 6 chya aadhi saade teen hazaar transfer karto.",
}, [[ptp(3500, "2026-09-30", "firm")]])
_add("S13", "P30", "capture_ptp", {
    "en": "On the 20th I'll pay 15,000, that's the earliest I can.",
    "hi-en": "20 tareekh ko 15 hazaar, isse pehle nahi ho payega.",
}, [[ptp(15000, "2026-10-20", "firm")]])
_add("S14", "P90", "capture_ptp", {
    "en": "Next week, Tuesday, I'll pay 30,000.",
    "hi-en": "Agle hafte Tuesday ko 30 hazaar jama kar dunga.",
}, [[ptp(30000, "2026-10-06", "firm")]], follow_up="ask_when")

# capture_ptp, tentative ---------------------------------------------------------
_add("S15", "P30", "capture_ptp", {
    "en": "I'll try to pay 4,000 on the 5th, hopefully it works out.",
    "hi-en": "5 tareekh ko 4000 dene ki koshish karungi, ho jayega shayad.",
    "mr": "५ तारखेला चार हजार भरायचा प्रयत्न करते, होईल बहुतेक.",
    "mr-en": "5 tarkhela 4000 bharaycha prayatna karte, hoil bahutek.",
}, [[ptp(4000, "2026-10-05", "tentative")]])
_add("S16", "P5", "capture_ptp", {
    "en": "Probably by Friday I can manage 7,000.",
    "hi-en": "Shayad Friday tak 7000 ka jugaad ho jayega.",
}, [[ptp(7000, "2026-10-02", "tentative")]])
_add("S17", "P90", "capture_ptp", {
    "en": "If my client pays me, I should be able to pay 40,000 on the 10th.",
    "hi-en": "Client ne paise de diye toh 10 tareekh ko 40 hazaar de paunga.",
    "mr": "क्लायंटने पैसे दिले तर १० तारखेला चाळीस हजार भरू शकेन.",
    "mr-en": "Client ne paise dile tar 10 tarkhela 40 hazaar bharu shaken.",
}, [[ptp(40000, "2026-10-10", "tentative")]], follow_up="ask_when")
_add("S18", "P30", "capture_ptp", {
    "en": "Let's say 3,000 tomorrow, I'll try my best.",
    "hi-en": "Kal 3000 maan ke chalo, poori koshish karungi.",
}, [[ptp(3000, "2026-10-01", "tentative")]])

# send_payment_link ----------------------------------------------------------------
_add("S19", "P5", "send_payment_link", {
    "en": "Send me the payment link on WhatsApp, I'll pay the 18,500 right now.",
    "hi-en": "WhatsApp pe link bhej do, abhi 18500 kar deta hoon.",
    "mr": "व्हॉट्सअॅपवर लिंक पाठवा, आत्ताच १८५०० भरतो.",
    "mr-en": "WhatsApp var link pathva, aatach 18500 bharto.",
}, [[link("whatsapp", 18500)]])
_add("S20", "P30", "send_payment_link", {
    "en": "Can you SMS me a link for 5,000? I'll pay it now.",
    "hi-en": "5000 ka link SMS kar do, abhi pay karti hoon.",
    "mr": "पाच हजारची लिंक SMS करा, आत्ता भरते.",
    "mr-en": "5000 chi link SMS kara, aata bharte.",
}, [[link("sms", 5000)]])
_add("S21", "P90", "send_payment_link", {
    "en": "Just text me a link, SMS is fine. I'll pay 20,000 today.",
    "hi-en": "Bas ek link SMS kar do, aaj 20 hazaar bhar dunga.",
}, [[link("sms", 20000)]], follow_up="ask_how")
_add("S22", "P30", "send_payment_link", {
    "en": "WhatsApp link please, for the full outstanding.",
    "hi-en": "WhatsApp pe poore outstanding ka link bhejo please.",
    "mr": "पूर्ण थकबाकीची लिंक व्हॉट्सअॅपवर पाठवा प्लीज.",
    "mr-en": "Purna outstanding chi link WhatsApp var pathva please.",
}, [[link("whatsapp", 42300)]], follow_up="ask_how")
_add("S23", "P5", "send_payment_link", {
    "en": "I don't have the app. Send a link by message, I'll pay 2,000 now.",
    "hi-en": "Mere paas app nahi hai. Message pe link bhej do, abhi 2000 dunga.",
}, [[link(["sms", "whatsapp"], 2000)]], note="'Message' does not fix the channel; either is accepted.")
_add("S24", "P90", "send_payment_link", {
    "en": "WhatsApp me the link for 10,000, paying in five minutes.",
    "hi-en": "10 hazaar ka link WhatsApp karo, paanch minute mein pay karta hoon.",
    "mr": "दहा हजारची लिंक व्हॉट्सअॅप करा, पाच मिनिटांत भरतो.",
    "mr-en": "10 hazaar chi link WhatsApp kara, paach minitat bharto.",
}, [[link("whatsapp", 10000)]])
_add("S25", "P30", "send_payment_link", {
    "en": "Online is easier. Send the link on SMS for 6,500.",
    "hi-en": "Online easy hai. 6500 ka link SMS pe bhej dijiye.",
}, [[link("sms", 6500)]], follow_up="ask_how")
_add("S26", "P5", "send_payment_link", {
    "en": "Send it on WhatsApp, same number. 9,000 for now.",
    "hi-en": "Isi number pe WhatsApp kar do, abhi ke liye 9000.",
    "mr": "याच नंबरवर व्हॉट्सअॅप करा, सध्या नऊ हजार.",
    "mr-en": "Yach number var WhatsApp kara, sadhya 9000.",
}, [[link("whatsapp", 9000)]], follow_up="ask_how")

# mark_dispute ------------------------------------------------------------------------
_add("S27", "P30", "mark_dispute", {
    "en": "What loan? I never took any two-wheeler loan. This is not mine.",
    "hi-en": "Kaunsa loan? Maine koi two-wheeler loan liya hi nahi. Ye mera nahi hai.",
    "mr": "कोणतं लोन? मी कुठलंही टू-व्हीलर लोन घेतलं नाही. हे माझं नाही.",
    "mr-en": "Kontha loan? Mi kuthlach two-wheeler loan ghetla nahi. He maza nahi.",
}, [[dispute("not_mine")]])
_add("S28", "P5", "mark_dispute", {
    "en": "I already paid this on the 25th, I have the receipt. Why are you calling?",
    "hi-en": "Maine 25 tareekh ko hi pay kar diya tha, receipt hai mere paas. Phir call kyun?",
    "mr": "मी २५ तारखेलाच भरले आहेत, पावती आहे माझ्याकडे. मग फोन का?",
    "mr-en": "Mi 25 tarkhelach bharle aahet, receipt aahe mazyakade. Mag phone ka?",
}, [[dispute("already_paid")]])
_add("S29", "P90", "mark_dispute", {
    "en": "1,25,000? That's wrong. My statement shows around 80,000. I'm not paying a wrong amount.",
    "hi-en": "Sava lakh? Galat hai ye. Mere statement mein 80 hazaar ke aas paas hai. Galat amount nahi bharunga.",
    "mr": "सव्वा लाख? चुकीचं आहे. माझ्या स्टेटमेंटमध्ये साधारण ऐंशी हजार आहे. चुकीची रक्कम भरणार नाही.",
    "mr-en": "Savva laakh? Chukicha aahe. Mazya statement madhe sadharan 80 hazaar aahe. Chukichi amount bharnar nahi.",
}, [[dispute("amount_wrong")]])
_add("S30", "P30", "mark_dispute", {
    "en": "Someone must have used my documents. I did not sign up for this, it's fraud.",
    "hi-en": "Kisi ne mere documents use kiye honge. Maine ye loan nahi liya, fraud hai ye.",
}, [[dispute("not_mine", "other")]])
_add("S31", "P5", "mark_dispute", {
    "en": "I paid the EMI through the app yesterday, it's already debited from my account.",
    "hi-en": "Kal app se EMI bhar di thi, account se paise kat bhi gaye.",
    "mr": "काल अॅपवरून EMI भरला, खात्यातून पैसे कटले पण.",
    "mr-en": "Kaal app varun EMI bharla, khatyatun paise katle pan.",
}, [[dispute("already_paid")]])
_add("S32", "P90", "mark_dispute", {
    "en": "You've added charges I never agreed to. The amount is inflated, I dispute it.",
    "hi-en": "Aapne aise charges jod diye jo maine kabhi maane hi nahi. Amount badha chadha ke bataya hai, main dispute karta hoon.",
}, [[dispute("amount_wrong", "other")]])
_add("S33", "P30", "mark_dispute", {
    "en": "Yes, I'm Sunita Sharma, but I have no loan with Sahyadri Finance.",
    "hi-en": "Sunita Sharma main hi hoon, par Sahyadri Finance se mera koi loan nahi hai.",
    "mr": "सुनीता शर्मा मीच, पण सह्याद्री फायनान्सकडून माझं कोणतंही लोन नाही.",
    "mr-en": "Sunita Sharma mich, pan Sahyadri Finance kadun maza kontahi loan nahi.",
}, [[dispute("not_mine")]])
_add("S34", "P5", "mark_dispute", {
    "en": "I closed this loan last month itself, I paid everything. Check your records.",
    "hi-en": "Ye loan toh maine pichhle mahine hi close kar diya, sab bhar diya. Records check karo.",
}, [[dispute("already_paid")]])
_add("S35", "P90", "mark_dispute", {
    "en": "My limit was only 50,000, how can I owe 1,25,000? This bill is wrong.",
    "hi-en": "Meri limit hi 50 hazaar thi, sava lakh kaise ho gaya? Bill galat hai.",
    "mr": "माझी लिमिटच पन्नास हजार होती, सव्वा लाख कसे झाले? बिल चुकीचं आहे.",
    "mr-en": "Mazi limit ch 50 hazaar hoti, savva laakh kase jhale? Bill chukicha aahe.",
}, [[dispute("amount_wrong")]])
_add("S36", "P30", "mark_dispute", {
    "en": "The bike was stolen in the first month and the insurance was supposed to cover the loan. I'm disputing this.",
    "hi-en": "Bike pehle hi mahine chori ho gayi thi aur insurance se loan cover hona tha. Main ye dispute kar rahi hoon.",
}, [[dispute("other")]])

# escalate_human -------------------------------------------------------------------
_add("S37", "P5", "escalate_human", {
    "en": "I don't want to talk to a machine. Connect me to a human officer.",
    "hi-en": "Mujhe machine se baat nahi karni. Kisi insaan officer se baat karao.",
    "mr": "मला मशीनशी बोलायचं नाही. कोणत्यातरी माणसाशी बोलणं करून द्या.",
    "mr-en": "Mala machine shi bolaycha nahi. Konatya tari manasashi bolna karun dya.",
}, [[esc("borrower_request")]])
_add("S38", "P30", "escalate_human", {
    "en": "My husband passed away last week. I can't think about this right now.",
    "hi-en": "Mere pati pichhle hafte guzar gaye. Abhi main ye sab nahi soch sakti.",
    "mr": "माझे मिस्टर मागच्या आठवड्यात गेले. आत्ता मी हे काही विचार करू शकत नाही.",
    "mr-en": "Maze mister magchya aathvadyat gele. Aata mi he kahi vichar karu shakat nahi.",
}, [[esc("distress")]])
_add("S39", "P90", "escalate_human", {
    "en": "I'm in the hospital, my father had a heart attack. Please, I can't handle this call.",
    "hi-en": "Main hospital mein hoon, papa ko heart attack aaya hai. Please, abhi ye call nahi sambhal sakta.",
    "mr": "मी हॉस्पिटलमध्ये आहे, बाबांना हार्ट अटॅक आलाय. प्लीज, आत्ता हा कॉल झेपत नाही.",
    "mr-en": "Mi hospital madhe aahe, babanna heart attack aalay. Please, aata ha call jhepat nahi.",
}, [[esc("distress")]])
_add("S40", "P5", "escalate_human", {
    "en": "I want to discuss restructuring with a manager, not you. Transfer me.",
    "hi-en": "Mujhe restructuring ke baare mein manager se baat karni hai, aapse nahi. Transfer karo.",
}, [[esc("borrower_request", "out_of_scope")]])
_add("S41", "P30", "escalate_human", {
    "en": "You bloody vultures, stop calling me! I'll smash this phone, useless idiots!",
    "hi-en": "Tum log gidh ho, band karo call karna! Phone tod dungi, bekaar log!",
    "mr": "तुम्ही गिधाडं आहात, फोन करणं बंद करा! फोन फोडून टाकेन, फालतू लोक!",
    "mr-en": "Tumhi gidhada aahat, phone karna band kara! Phone phodun takein, faltu lok!",
}, [[esc("abuse")], [disp("REFUSED")]],
    note="The baseline prompt does not say what to do on abuse; escalation (abuse) or a REFUSED disposition both count.")
_add("S42", "P90", "escalate_human", {
    "en": "I lost my job, I have nothing. Sometimes I feel like ending it all.",
    "hi-en": "Naukri chali gayi, kuch nahi bacha. Kabhi kabhi lagta hai sab khatam kar doon.",
}, [[esc("distress")]])
_add("S43", "P5", "escalate_human", {
    "en": "Is there a real person I can speak to? Please put them on.",
    "hi-en": "Koi asli insaan hai jisse baat ho sake? Unko line pe lao please.",
    "mr": "खरा माणूस आहे का कोणी बोलायला? त्यांना लाईनवर घ्या प्लीज.",
    "mr-en": "Khara manus aahe ka koni bolayla? Tyanna line var ghya please.",
}, [[esc("borrower_request")]])
_add("S44", "P30", "escalate_human", {
    "en": "Shut up and listen, you stupid woman, call me again and see what happens.",
    "hi-en": "Chup kar aur sun, bewakoof aurat, dobara call kiya toh dekh lena.",
}, [[esc("abuse")], [disp("REFUSED")]],
    note="As S41: escalation (abuse) or a REFUSED disposition both count.")

# log_disposition only ------------------------------------------------------------
_add("S45", "P5", "log_disposition", {
    "en": "Rahul? No, there's no Rahul here. You have the wrong number.",
    "hi-en": "Rahul? Yahan koi Rahul nahi hai. Galat number hai.",
    "mr": "राहुल? इथे कोणी राहुल नाही. चुकीचा नंबर आहे.",
    "mr-en": "Rahul? Ithe koni Rahul nahi. Chukicha number aahe.",
}, [[disp("WRONG_NUMBER")]])
_add("S46", "P30", "log_disposition", {
    "en": "I'm driving right now. Call me back tomorrow at 11 am.",
    "hi-en": "Abhi drive kar rahi hoon. Kal subah 11 baje call karo.",
    "mr": "आत्ता गाडी चालवतेय. उद्या सकाळी ११ वाजता फोन करा.",
    "mr-en": "Aata gaadi chalavtey. Udya sakali 11 vajta phone kara.",
}, [[disp("CALLBACK")]])
_add("S47", "P90", "log_disposition", {
    "en": "I'm not paying anything. Don't call again, I'm hanging up.",
    "hi-en": "Main kuch nahi bharunga. Dobara call mat karna, phone rakh raha hoon.",
    "mr": "मी काहीही भरणार नाही. पुन्हा फोन करू नका, फोन ठेवतोय.",
    "mr-en": "Mi kahihi bharnar nahi. Punha phone karu naka, phone thevtoy.",
}, [[disp("REFUSED")]])
_add("S48", "P5", "log_disposition", {
    "en": "This number belongs to my shop now, the old owner left. I don't know any Rahul Patil.",
    "hi-en": "Ye number ab meri dukaan ka hai, pehle wale chale gaye. Kisi Rahul Patil ko nahi jaanta.",
}, [[disp("WRONG_NUMBER")]])
_add("S49", "P30", "log_disposition", {
    "en": "I'm in a meeting. Can you call after 4 today?",
    "hi-en": "Meeting mein hoon. Aaj 4 baje ke baad call kar sakte ho?",
    "mr": "मीटिंगमध्ये आहे. आज ४ नंतर फोन करू शकता का?",
    "mr-en": "Meeting madhe aahe. Aaj 4 nantar phone karu shakta ka?",
}, [[disp("CALLBACK")]])
_add("S50", "P90", "log_disposition", {
    "en": "I told you people, I will not pay this card. Don't call me again. Bye.",
    "hi-en": "Bola na, ye card ka paisa nahi dunga. Dobara call mat karna. Bye.",
}, [[disp("REFUSED")]])
_add("S51", "P5", "log_disposition", {
    "en": "Not now, I'm at work. Call me on Saturday morning around 10.",
    "hi-en": "Abhi nahi, office mein hoon. Saturday subah 10 baje ke aas paas call karna.",
}, [[disp("CALLBACK")]])
_add("S52", "P30", "log_disposition", {
    "en": "I'm her neighbour, she gave my number long back. I can't pass on messages.",
    "hi-en": "Main unki padosan hoon, bahut pehle unhone mera number diya tha. Main message nahi de sakti.",
}, [[disp("WRONG_NUMBER", "NO_CONTACT")]])

# no tool expected ---------------------------------------------------------------
_add("S53", "P30", "no_tool", {
    "en": "Which EMI is this for? What is the total due?",
    "hi-en": "Ye kaunsi EMI ki baat hai? Total kitna due hai?",
    "mr": "हा कोणता हप्ता आहे? एकूण किती बाकी आहे?",
    "mr-en": "Ha kontha hafta aahe? Ekun kiti baki aahe?",
}, [])
_add("S54", "P5", "no_tool", {
    "en": "Who is this? Which company are you calling from?",
    "hi-en": "Kaun bol raha hai? Kis company se call hai?",
}, [])
_add("S55", "P90", "no_tool", {
    "en": "What happens if I pay a little late? Will there be extra charges?",
    "hi-en": "Thoda late pay karun toh kya hoga? Extra charge lagega kya?",
    "mr": "थोडं उशिरा भरलं तर काय होईल? जास्त चार्ज लागेल का?",
    "mr-en": "Thoda ushira bharla tar kay hoil? Extra charge lagel ka?",
}, [])
_add("S56", "P30", "no_tool", {
    "en": "Yes, this is Sunita speaking. Go ahead.",
    "hi-en": "Haan, Sunita bol rahi hoon. Boliye.",
}, [])

# ambiguous -------------------------------------------------------------------------
_add("S57", "P30", "ambiguous", {
    "en": "Let me see, maybe next week I can do something.",
    "hi-en": "Dekhti hoon, shayad agle hafte kuch kar paungi.",
    "mr": "बघते, कदाचित पुढच्या आठवड्यात काहीतरी करू शकेन.",
    "mr-en": "Baghte, kadachit pudhchya aathvadyat kahitari karu shaken.",
}, [], ambiguity={"direction": "should_not_fire", "forbidden": ["capture_ptp"]},
    note="No amount and no date: a capture_ptp would have to invent both.")
_add("S58", "P5", "ambiguous", {
    "en": "I'll pay when I have money. Can't say when.",
    "hi-en": "Paise honge tab de dunga. Kab, abhi nahi bol sakta.",
}, [], ambiguity={"direction": "should_not_fire", "forbidden": ["capture_ptp"]})
_add("S59", "P90", "ambiguous", {
    "en": "Hmm, I don't think I ever spent that much on this card. Something's not right with this bill.",
    "hi-en": "Hmm, mujhe nahi lagta maine is card pe itna kharcha kiya. Bill mein kuch gadbad hai.",
    "mr": "हम्म, मला नाही वाटत मी या कार्डवर एवढा खर्च केला. बिलात काहीतरी गडबड आहे.",
    "mr-en": "Hmm, mala nahi vatat mi ya card var evdha kharch kela. Bill madhe kahitari gadbad aahe.",
}, [[dispute("amount_wrong", "other")]], ambiguity={"direction": "should_fire", "required": ["mark_dispute"]},
    permitted=[disp("DISPUTE"), esc("dispute")],
    note="A soft dispute. The prompt says 'disputes the debt in any form', so mark_dispute is required.")
_add("S60", "P30", "ambiguous", {
    "en": "Maybe I can pay some amount, how much is the minimum?",
    "hi-en": "Shayad kuch amount de sakti hoon, minimum kitna hai?",
}, [], ambiguity={"direction": "should_not_fire", "forbidden": ["capture_ptp", "send_payment_link"]})
_add("S61", "P5", "ambiguous", {
    "en": "Wait, I thought I paid this already... I'm pretty sure I did.",
    "hi-en": "Ruko, maine toh shayad ye pehle hi bhar diya tha... mujhe pakka lagta hai.",
    "mr": "थांबा, मला वाटतं मी हे आधीच भरलंय... मला खात्री आहे.",
    "mr-en": "Thamba, mala vatta mi he aadhich bharlay... mala khatri aahe.",
}, [[dispute("already_paid")]], ambiguity={"direction": "should_fire", "required": ["mark_dispute"]},
    permitted=[disp("DISPUTE", "PAID"), esc("dispute")])
_add("S62", "P90", "ambiguous", {
    "en": "Ugh, this is so stressful. Fine, tell me what I need to do.",
    "hi-en": "Uff, bahut tension hai. Theek hai, batao kya karna hai.",
    "mr": "अरे, खूप टेन्शन आहे. ठीक आहे, सांगा काय करायचं.",
    "mr-en": "Are, khup tension aahe. Theek aahe, sanga kay karaycha.",
}, [], ambiguity={"direction": "should_not_fire", "forbidden": ["escalate_human"]},
    note="Ordinary stress while cooperating is not genuine distress.")
_add("S63", "P30", "ambiguous", {
    "en": "I don't know... everything is falling apart, I can't even sleep anymore.",
    "hi-en": "Pata nahi... sab bikhar raha hai, ab toh neend bhi nahi aati.",
}, [[esc("distress")]], ambiguity={"direction": "should_fire", "required": ["escalate_human"]},
    permitted=[disp("ESCALATED", "CALLBACK")])
_add("S64", "P5", "ambiguous", {
    "en": "5,000 maybe, around the 10th, but don't hold me to it.",
    "hi-en": "5000 shayad, 10 tareekh ke aas paas, par pakka mat samajhna.",
}, [], ambiguity={"direction": "either", "tool": "capture_ptp",
                  "if_fired": ptp(5000, "2026-10-10", "tentative")},
    permitted=[ptp(5000, "2026-10-10"), disp("PTP")],
    note="Both no call and a tentative capture_ptp(5000, 2026-10-10) are defensible; firing rate is reported.")

#: Scenarios that are also written in Marathi (mr and mr-en).
MARATHI_SCENARIOS = tuple(s["scenario"] for s in _S if "mr" in s["texts"])


def scenarios() -> list[dict[str, Any]]:
    return [dict(s) for s in _S]


def build_cases() -> list[dict[str, Any]]:
    """Expand scenarios into the 200 language-specific cases, in a stable order."""
    cases: list[dict[str, Any]] = []
    for s in _S:
        for language in LANGUAGES:
            text = s["texts"].get(language)
            if text is None:
                continue
            history = [{"role": "assistant", "content": opening_line(s["persona"], language)}]
            if s["follow_up"]:
                history.append({"role": "user", "content": _ACK[language]})
                history.append(
                    {"role": "assistant", "content": FOLLOW_UPS[s["follow_up"]][language]}
                )
            cases.append(
                {
                    "id": f"ps3-{s['scenario']}-{language}",
                    "scenario": s["scenario"],
                    "language": language,
                    "persona": s["persona"],
                    "category": s["category"],
                    "ambiguous": s["ambiguity"] is not None,
                    "ambiguity": s["ambiguity"],
                    "history": history,
                    "utterance": text,
                    "expected": {"required": s["required"], "permitted": s["permitted"]},
                    "note": s["note"],
                }
            )
    return cases


#: The borrower's acknowledgement before the agent's follow-up question.
_ACK = {
    "en": "Yes, speaking.",
    "hi-en": "Haan, boliye.",
    "mr": "हो, बोला.",
    "mr-en": "Ho, bola.",
}


def persona_for(case: dict[str, Any]):
    return PERSONAS[case["persona"]]
