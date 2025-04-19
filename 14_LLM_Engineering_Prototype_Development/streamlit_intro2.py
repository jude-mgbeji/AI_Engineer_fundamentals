import streamlit as st

with st.chat_message("user"):
    st.write("Hello, how are you?")

with st.chat_message("AI"):
    st.write("I'm fine, thank you! How can I assist you today?")

prompt = st.chat_input("Type a message...", max_chars=50)

if prompt:
    with st.chat_message("user"):
        st.write(prompt)