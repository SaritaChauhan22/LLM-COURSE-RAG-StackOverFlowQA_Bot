import os
import pandas as pd

from config import *

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline



class RAGChatbot:


    def __init__(self):

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )


        # Create / Load FAISS
        self.vectorstore = self.create_vectorstore()


        # Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k":3
            }
        )


        # LLM
        pipe = pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            max_length=256
        )


        self.llm = HuggingFacePipeline(
            pipeline=pipe
        )



    # -------------------------
    # Load CSV
    # -------------------------
    def load_documents(self):

        df = pd.read_csv(
            CSV_PATH,
            encoding="utf-8",
            sep=",",
            quotechar='"',
            engine="python",
            on_bad_lines="skip"
        )


        documents = []


        for _, row in df.iterrows():

            question = str(row.get("Question", ""))

            answer = str(row.get("Answer", ""))


            text = f"""
    Question:
    {question}

    Answer:
    {answer}
    """


            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "question": question
                    }
                )
            )


        return documents


    # -------------------------
    # Create FAISS
    # -------------------------

    def create_vectorstore(self):


        if os.path.exists(VECTOR_DB):

            return FAISS.load_local(

                VECTOR_DB,

                self.embeddings,

                allow_dangerous_deserialization=True

            )


        documents = self.load_documents()


        splitter = RecursiveCharacterTextSplitter(

            chunk_size=800,

            chunk_overlap=100

        )


        chunks = splitter.split_documents(
            documents
        )


        db = FAISS.from_documents(

            chunks,

            self.embeddings

        )


        db.save_local(
            VECTOR_DB
        )


        return db




    # -------------------------
    # Ask Question
    # -------------------------

    def ask(self, question):


        # Retrieve documents

        docs = self.retriever.invoke(
            question
        )


        if not docs:

            return "I don't know based on documents."



        context = "\n\n".join(

            [
                doc.page_content

                for doc in docs
            ]

        )


        prompt=f"""

You are a Stack Overflow assistant.

Use only the context below.

Context:

{context}


Question:

{question}


Answer in simple complete sentences:

"""


        answer = self.llm.invoke(
            prompt
        )


        return answer