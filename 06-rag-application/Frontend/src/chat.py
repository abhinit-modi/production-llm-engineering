import streamlit as st
from api import get_response

def chat_interface(chat_title, page_hash ,url):
    st.title(chat_title)

    # Add username input at the top of the page
    username = st.text_input("Enter your username:", key="username_input", value="Guest")

    # Initialize page-specific chat history
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    
    if page_hash not in st.session_state.chat_histories:
        st.session_state.chat_histories[page_hash] = []

    # Display chat messages from history for the current page
    for message in st.session_state.chat_histories[page_hash]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your message?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.chat_histories[page_hash].append({"role": "user", "content": prompt})

        # Get streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in get_response(prompt, url, username):
                full_response += chunk
                response_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.chat_histories[page_hash].append({"role": "assistant", "content": full_response})