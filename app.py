import streamlit as st
from rag import RAGChatbot

st.set_page_config(
    page_title="Mini RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 StackOverflow QA RAG Chatbot")

st.write(
    "CSV → Documents → Chunks → Embeddings → FAISS → Retriever → Hugging Face"
)


@st.cache_resource
def load_bot():
    return RAGChatbot()


bot = load_bot()


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("Ask a StackOverflow question...")


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:
                answer = bot.ask(question)
            except Exception as e:
                answer = f"❌ Error:\n\n{e}"

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )