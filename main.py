import os

import streamlit as st
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine

qa_pairs = [
        {
            "question": "What does the eligibility verification agent (EVA) do?",
            "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
        },
        {
            "question": "What does the claims processing agent (CAM) do?",
            "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
        },
        {
            "question": "How does the payment posting agent (PHIL) work?",
            "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
        },
        {
            "question": "Tell me about Thoughtful AI's Agents.",
            "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        },
        {
            "question": "What are the benefits of using Thoughtful AI's agents?",
            "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
        }
    ]

def load_data():
    docs = [Document(doc_id=f"doc_{i}", text=f"Question: {pair['question']}\nAnswer: {pair['answer']}") for i, pair in enumerate(qa_pairs)]
    index = VectorStoreIndex.from_documents(docs)
    return index
    
def get_chat_engine():
    index = load_data()           
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm)
    return chat_engine

def main():
    title = "Thoughtful AI customer support agent"
    st.set_page_config(page_title=title, layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title(title)
    st.session_state.chat_engine = get_chat_engine()
    
    if "messages" not in st.session_state.keys() or len(st.session_state.messages)==0: # Initialize the chat messages history
        st.session_state.messages = []
        
    
    prompt = ""

    if not prompt:
        # Prompt for user input and save to chat history
        prompt = st.chat_input("Your question")
    if prompt: 
        st.session_state.messages.append({"role": "user", "content": prompt})
        
    # Display the prior chat messages
    for message in st.session_state.messages: 
        with st.chat_message(message["role"]):
            st.write(message["content"])
        
    if st.session_state.messages:
        message = st.session_state.messages[-1]
        if message["role"] != "assistant":
            with st.chat_message("assistant"):
                response_container = st.empty()  # Container to hold the response as it streams
                response_msg = ""
                
                if prompt:
                    streaming_response = st.session_state.chat_engine.stream_chat(prompt)
                else:
                    st.rerun()
                
                for token in streaming_response.response_gen:
                    response_msg += token
                    response_container.write(response_msg)
                    
                message = {"role": "assistant", "content": response_msg}
                st.session_state.messages.append(message) # Add response to message history
                    
main()