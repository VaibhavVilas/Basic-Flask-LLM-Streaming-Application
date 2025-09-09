from flask import Flask, Response, send_file, request
from flask_cors import CORS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Your own OpenAI API Key

# Initialize Flask app
app = Flask(__name__)

# Add CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "*"]}})

# Initialize embedding function
embedding_function = OpenAIEmbeddings()

# Define sample documents
docs = [
    Document(
        page_content="""
        LangChain is built around the idea of chaining operations. At its core, it’s a framework for executing a sequence of functions in a chain. Think of it as a pipeline where each step depends on the output of the previous one.For example, imagine you’re building an application that needs to retrieve data from a website, summarize it, and then answer user questions based on that summary. LangChain helps you break this down into three steps: retrieve, summarize, and answer.
To retrieve data, you might use a LangChain component called a document loader. This fetches content from various sources. If the documents are large, you might use a text splitter to break them into smaller, meaningful chunks.For summarization, you’d use a chain that orchestrates the process. This might involve constructing a prompt to instruct the LLM and passing the request to the model. The answer step would involve another chain, possibly with a memory component to store conversation history and context, along with another prompt and LLM to generate the final response. One of the strengths of LangChain is its modularity. You can mix and match components to build complex workflows. For instance, the LLM used for answering questions might be entirely different from the one used for summarization. This flexibility makes it a great choice for applications where you know the exact sequence of steps needed.
        """, metadata={"source": "langchain.txt"}
    ),
    Document(
        page_content="""
        LangGraph, on the other hand, is designed for more complex, stateful workflows. It’s a specialized library within the LangChain ecosystem, tailored for building multi-agent systems that handle nonlinear processes.Consider a task management assistant. The workflow here isn’t linear. It involves processing user input, adding tasks, completing tasks, and summarizing tasks. LangGraph models this as a graph structure, where each action is a node and the transitions between actions are edges.The central node is the process input node, where user input is received and routed to the appropriate action node. There’s also a state component that maintains the task list across interactions. Nodes like “add task” and “complete task” modify this state, while the “summarize” node generates an overview of current tasks.Traditional cybersecurity methods aren’t enough. Models can be poisoned, stolen, or tricked in ways traditional software can’t. If you’re building AI, you need a new approach.The graph structure allows for loops and revisiting previous states, making it ideal for interactive systems where the next step depends on evolving conditions or user input. This flexibility is what sets LangGraph apart. It’s designed for applications that need to maintain context over extended interactions, like virtual assistants or complex task management systems.
        """, metadata={"source": "langgraph.txt"}
    ),
]

# Initialize Chroma vector store and add documents
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_function,
)
uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)

# Set up retriever
retriever = vector_store.as_retriever()

# Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize model
model = ChatOpenAI(temperature=0, streaming=True)

# Set up retrieval chain
retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# # Streaming response generator
# def generate_chat_responses(message):
#     for chunk in retrieval_chain.stream(message):  # Use synchronous stream
#         content = chunk.replace("\n", "<br>")
#         yield f"data: {content}\n\n"

# # Root endpoint to serve HTML
# @app.route("/")
# def root():
#     return send_file("static/index.html")

# # Chat streaming endpoint
# @app.route("/chat_stream/<path:message>")
# def chat_stream(message):
#     return Response(generate_chat_responses(message), mimetype="text/event-stream")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, threaded=True)

# Dictionary to track active streaming sessions
active_streams = {}

# Streaming response generator
def generate_chat_responses(message, session_id):
    try:
        for chunk in retrieval_chain.stream(message):
            # Check if session has been cancelled
            if session_id in active_streams and active_streams[session_id]["cancelled"]:
                break
            content = chunk.replace("\n", "<br>")
            yield f"data: {content}\n\n"
    finally:
        # Clean up session
        if session_id in active_streams:
            del active_streams[session_id]

# Chat streaming endpoint
@app.route("/chat_stream/<path:message>")
def chat_stream(message):
    session_id = request.args.get("session_id")
    if not session_id:
        return {"error": "Session ID required"}, 400

    # Mark session as active
    active_streams[session_id] = {"cancelled": False}

    return Response(generate_chat_responses(message, session_id), mimetype="text/event-stream")

# Stop stream endpoint
@app.route("/stop_stream/<session_id>", methods=["POST"])
def stop_stream(session_id):
    if session_id in active_streams:
        print("active_streams",active_streams)
        active_streams[session_id]["cancelled"] = True
        return {"status": "Stream stopped"}, 200
    return {"error": "Session not found"}, 404

# Root endpoint (unchanged)
@app.route("/")
def root():
    return send_file("static/index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)