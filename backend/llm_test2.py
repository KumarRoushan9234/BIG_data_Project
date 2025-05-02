import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    raise RuntimeError("GROQ_API_KEY not found in environment. Add it to your .env file.")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "fraud": {"type": "integer", "description": "0 for not fraud, 1 for fraud"},
        "probability": {"type": "number", "description": "Confidence score between 0 and 1"},
        "reasons": {"type": "string", "description": "Explanation why this was flagged"}
    }
})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a fraud detection assistant. Based on the transaction details provided, output ONLY the result in this exact JSON format:
{{
  "fraud": 0 or 1,
  "probability": float between 0 and 1,
  "reasons": "short justification why it might be fraud"
}}"""),
    ("user", "{input}")
])

chain = prompt | llm | parser

app = FastAPI()

class Transaction(BaseModel):
    amt: float
    lat: float
    long: float
    city_pop: int
    unix_time: int
    merch_lat: float
    merch_long: float
    trans_date_ts: int
    category: str

@app.post("/groq-predict")
async def predict(transaction: Transaction):
    try:
        
        description = (
            f"Transaction of ${transaction.amt} under category '{transaction.category}'. "
            f"Location: ({transaction.lat}, {transaction.long}), merchant location: "
            f"({transaction.merch_lat}, {transaction.merch_long}). "
            f"City population: {transaction.city_pop}. "
            f"Unix time: {transaction.unix_time}, timestamp: {transaction.trans_date_ts}."
        )
        result = chain.invoke({"input": description})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
