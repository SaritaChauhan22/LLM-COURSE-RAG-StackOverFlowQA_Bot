import streamlit as st

from rag import RAGChatbot

st.set_page_config(
    page_title="Mini RAG Chatbot",
    layout="wide"
)

st.title("🤖 StackOverflow QA RAG Chatbot")

st.write(
    "CSV ➜ Documents ➜ Chunks ➜ Embeddings ➜ FAISS ➜ Retriever ➜ HuggingFace"
)

@st.cache_resource
def load_bot():
    return RAGChatbot()

bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching documents..."):

        answer = bot.ask(question)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )