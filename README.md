# Basic-Flask-LLM-Streaming-Application

A Simple Flask applicaiton to showcase LLM streaming by using langchain and also utlizing simple Retrieval-Augmented Generation(RAG) via Chroma Vector store. This Web Application also provides a "STOP" button, to stop generating response before completing, hence showcasing the advantage of streaming allowing us to save tokens as well.

# Prerequisite - 
OPENAI API KEY

# Steps to run application -

1. Create a virtual env using "python -m venv name-of-env" and activate the same.
2. git clone git@github.com:VaibhavVilas/Basic-Flask-LLM-Streaming-Application.git
3. cd Basic-Flask-LLM-Streaming-Application 
4. pip install -r requirements.txt 
5. python app.py
6. Go to "http://127.0.0.1:5000/"
7. The implementation contains data on Langchain,Langgraph, stored in Chroma. So, query should be around that; e.g - "Tell me about langchain in about 300 words".
8. You can replace/add data as per your requirements
