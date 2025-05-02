import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables from .env file
load_dotenv()

# Ensure GROQ_API_KEY is available
if "GROQ_API_KEY" not in os.environ:
    raise RuntimeError("GROQ_API_KEY not found in environment. Add it to your .env file.")

# Initialize the LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",  # You can use "llama-3.1-8b-instant" if available
    temperature=0.2
)

# Define the output parser
parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "fraud": {
            "type": "integer",
            "description": "0 for not fraud, 1 for fraud"
        },
        "probability": {
            "type": "number",
            "description": "Confidence score between 0 and 1"
        },
        "reasons": {
            "type": "string",
            "description": "Explanation why this was flagged"
        }
    }
})

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a fraud detection assistant. Based on the transaction details provided, output ONLY the result in this exact JSON format:
{
  "fraud": 0 or 1,
  "probability": float between 0 and 1,
  "reasons": "short justification why it might be fraud"
}"""),
    ("user", "{input}")
])

# Chain the prompt, model, and parser
chain = prompt | llm | parser
