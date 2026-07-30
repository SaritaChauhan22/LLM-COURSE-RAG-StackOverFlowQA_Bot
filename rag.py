# import os
# import pandas as pd

# from config import *

# from langchain_core.documents import Document

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.vectorstores import FAISS

# from transformers import pipeline

# from langchain_huggingface import HuggingFacePipeline



# class RAGChatbot:


#     def __init__(self):

#         # Embeddings
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL
#         )


#         # Create / Load FAISS
#         self.vectorstore = self.create_vectorstore()


#         # Retriever
#         self.retriever = self.vectorstore.as_retriever(
#             search_kwargs={
#                 "k":3
#             }
#         )


#         # LLM
#         pipe = pipeline(
#             "text2text-generation",
#             model=LLM_MODEL,
#             max_length=256
#         )


#         self.llm = HuggingFacePipeline(
#             pipeline=pipe
#         )



#     # -------------------------
#     # Load CSV
#     # -------------------------
#     def load_documents(self):

#         df = pd.read_csv(
#             CSV_PATH,
#             encoding="utf-8",
#             sep=",",
#             quotechar='"',
#             engine="python",
#             on_bad_lines="skip"
#         )


#         documents = []


#         for _, row in df.iterrows():

#             question = str(row.get("Question", ""))

#             answer = str(row.get("Answer", ""))


#             text = f"""
#     Question:
#     {question}

#     Answer:
#     {answer}
#     """


#             documents.append(
#                 Document(
#                     page_content=text,
#                     metadata={
#                         "question": question
#                     }
#                 )
#             )


#         return documents


#     # -------------------------
#     # Create FAISS
#     # -------------------------

#     def create_vectorstore(self):


#         if os.path.exists(VECTOR_DB):

#             return FAISS.load_local(

#                 VECTOR_DB,

#                 self.embeddings,

#                 allow_dangerous_deserialization=True

#             )


#         documents = self.load_documents()


#         splitter = RecursiveCharacterTextSplitter(

#             chunk_size=800,

#             chunk_overlap=100

#         )


#         chunks = splitter.split_documents(
#             documents
#         )


#         db = FAISS.from_documents(

#             chunks,

#             self.embeddings

#         )


#         db.save_local(
#             VECTOR_DB
#         )


#         return db




#     # -------------------------
#     # Ask Question
#     # -------------------------

#     def ask(self, question):


#         # Retrieve documents

#         docs = self.retriever.invoke(
#             question
#         )


#         if not docs:

#             return "I don't know based on documents."



#         context = "\n\n".join(

#             [
#                 doc.page_content

#                 for doc in docs
#             ]

#         )


#         prompt=f"""

# You are a Stack Overflow assistant.

# Use only the context below.

# Context:

# {context}


# Question:

# {question}


# Answer in simple complete sentences:

# """


#         answer = self.llm.invoke(
#             prompt
#         )


#         return answer



import pandas as pd
import os

from transformers import pipeline

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================
# CONFIGURATION
# ==========================

CSV_PATH = "StackOverflow_QA_Format.csv"

VECTOR_DB = "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

LLM_MODEL = "google/flan-t5-small"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 3



class RAGChatbot:

    def __init__(self):

        # ==========================
        # LOAD CSV DATA
        # ==========================

        df = pd.read_csv(
            CSV_PATH,
            encoding="utf-8",
            on_bad_lines="skip"
        )


        documents = []

        for _, row in df.iterrows():

            text = ""

            for col in df.columns:
                text += f"{col}: {row[col]}\n"


            documents.append(
                Document(
                    page_content=text
                )
            )


        # ==========================
        # CHUNKING
        # ==========================

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )


        chunks = splitter.split_documents(documents)



        # ==========================
        # EMBEDDINGS
        # ==========================

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )


        # ==========================
        # FAISS VECTOR DATABASE
        # ==========================

        if os.path.exists(VECTOR_DB):

            self.vectorstore = FAISS.load_local(
                VECTOR_DB,
                embeddings,
                allow_dangerous_deserialization=True
            )

        else:

            self.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            self.vectorstore.save_local(
                VECTOR_DB
            )


        # ==========================
        # RETRIEVER
        # ==========================

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )


        # ==========================
        # HUGGINGFACE LLM
        # ==========================

        self.llm = pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            max_new_tokens=256
        )



    def ask(self, question):

        docs = self.retriever.invoke(question)


        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )


        prompt = f"""
You are a helpful StackOverflow assistant.

Use the context below to answer.

Context:
{context}

Question:
{question}

Answer:
"""


        response = self.llm(prompt)


        return response[0]["generated_text"]