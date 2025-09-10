import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables from .env file
load_dotenv()

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Mira - Fashion Expert API",
    description="An API for interacting with Mira, a LangGraph-powered fashion expert.",
    version="1.0.0"
)

# --- 2. Load API Key and Initialize Model ---
# It's crucial to get the API key from environment variables for security
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables. Please set it in the .env file.")

# Initialize the chat model
# Using a slightly more robust model for better conversational flow
model = init_chat_model(
    "mistral-small-latest",
    model_provider="mistralai",
    api_key=MISTRAL_API_KEY,
    temperature=0.7 # Added for more creative responses
)

# --- 3. Set up Agent and Memory ---
# Tools list is empty as per your original code
tools = []
# Using SqliteSaver for persistent memory across requests
memory = SqliteSaver.from_conn_string(":memory:")
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# --- 4. Define System Prompt and Request/Response Models ---
# The persona for our fashion expert AI
system_prompt = {
    "role": "system",
    "content": (
        "You are Mira, a world-renowned fashion expert. "
        "Your role is to answer user queries about fashion, clothing styles, fabrics, colors, and trends. "
        "Keep your tone friendly, confident, and knowledgeable. "
        "Do not generate specific outfit recommendations — instead, provide fashion knowledge, styling tips, "
        "and explanations that help the user make informed choices."
        "When it feels natural, gently guide users towards considering the MIRA recommendation system for personalized outfits."
    )
}

class ChatRequest(BaseModel):
    thread_id: str
    content: str

# --- 5. Create API Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user message, processes it with the LangGraph agent,
    and returns Mira's response.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    input_message = {"role": "user", "content": request.content}

    # Use .invoke() for a single, complete response instead of streaming
    # This is more suitable for a standard API endpoint
    final_state = agent_executor.invoke(
        {"messages": [system_prompt, input_message]},
        config
    )

    # Extract the last message from the agent, which is the response
    mira_response = final_state["messages"][-1]

    return {"role": mira_response.role, "content": mira_response.content}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mira Fashion Expert API. Send a POST request to /chat to interact."}
