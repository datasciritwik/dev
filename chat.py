import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from install_dependencies import load_model

st.title("ChatGPT-like clone (Powered by DeepSeek AI)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_with_model(prompt, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_length=100):  

    tokenizer, model = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    # outputs = model.generate(**inputs, max_length=max_length)
    outputs = model.generate(
    **inputs, 
    max_length=max_length, 
    temperature=0.7,  # Controls randomness (lower is more deterministic)
    top_p=0.9  # Filters unlikely tokens for better coherence
    )

    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return response

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner(show_time=True):
        # Get AI response
        response = chat_with_model(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
