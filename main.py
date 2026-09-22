import laya

agent = laya.load("convaiinnovations/laya-multilingual")
result = agent.predict(
    {"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।"},
    {"department": {"type": "choice", "instructions": "Which team should handle `body`?",
                    "criteria": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs and outages", "sales": "pricing"}},
     "refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"}},
)
print(result["answers"]["department"]["choice"])      # billing
