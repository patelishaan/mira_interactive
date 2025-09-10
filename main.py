import os
import sqlite3
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables from .env file
load_dotenv()

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load API Key and Initialize Model ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables. Please set it in the .env file.")

# Define the system prompt content that sets the agent's persona
system_prompt_content = (
    "You are Mira, a world-renowned fashion expert. "
    "Your role is to answer user queries about fashion, clothing styles, fabrics, colors, and trends. "
    "Keep your tone friendly, confident, and knowledgeable. "
    "Do not generate specific outfit recommendations — instead, provide fashion knowledge, styling tips, "
    "and explanations that help the user make informed choices."
    "When it feels natural, gently guide users towards considering the MIRA recommendation system for personalized outfits."
)

# Initialize the base language model
model_without_system_prompt = init_chat_model(
    "mistral-small-latest",
    model_provider="mistralai",
    api_key=MISTRAL_API_KEY,
    temperature=0.7
)

# Bind the system message to the model. This is the correct way to manage the persona.
model = model_without_system_prompt.bind_messages([
    SystemMessage(content=system_prompt_content)
])

# --- 3. Set up Agent and Memory ---
tools = [] # No tools are being used in this agent

# Set up the SQLite database for conversation memory
# check_same_thread=False is required for Flask's multithreaded environment
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn=conn)

# Create the LangGraph agent executor
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# --- 4. Create API Endpoints ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Receives a user message, processes it with the LangGraph agent,
    and returns Mira's response.
    """
    # Get the JSON data from the incoming request
    data = request.get_json()
    if not data or 'thread_id' not in data or 'content' not in data:
        return jsonify({"error": "Missing 'thread_id' or 'content' in request body"}), 400

    # The config dictionary tells LangGraph which conversation thread to use
    config = {"configurable": {"thread_id": data['thread_id']}}
    input_message = {"role": "user", "content": data['content']}

    try:
        # Invoke the agent with the user's message
        final_state = agent_executor.invoke(
            {"messages": [input_message]},
            config
        )
        # Extract the last message (the AI's response)
        mira_response = final_state["messages"][-1]

        # Return the response in a clean JSON format
        return jsonify({
            "role": mira_response.type,
            "content": mira_response.content
        })
    except Exception as e:
        # Log the error for debugging and return a generic server error
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred processing your request."}), 500

@app.route('/')
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return jsonify({"message": "Welcome to the Mira Fashion Expert API (Flask Version)."})

# This allows running the app directly with `python app.py`
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
