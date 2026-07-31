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

import os
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===================================================
# CONFIG
# ===================================================

CSV_PATH = "StackOverflow_QA_Format.csv"

VECTOR_DB = "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "google/flan-t5-small"

TOP_K = 3

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


def load_seq2seq_model(model_name: str = LLM_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


class SimpleRetriever:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.embedded_documents = []

        for doc in documents:
            embedding = np.asarray(self.embeddings.embed_query(doc.page_content), dtype=np.float32)
            self.embedded_documents.append(embedding)

        if self.embedded_documents:
            self.embedded_documents = np.vstack(self.embedded_documents)
        else:
            self.embedded_documents = np.empty((0, 0), dtype=np.float32)

    def invoke(self, query):
        if self.embedded_documents.size == 0:
            return []

        query_embedding = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)

        if self.embedded_documents.ndim != 2 or query_embedding.ndim != 1:
            return self.documents[:TOP_K]

        if self.embedded_documents.shape[1] != query_embedding.shape[0]:
            return self.documents[:TOP_K]

        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-12:
            return self.documents[:TOP_K]

        query_embedding = query_embedding / query_norm
        doc_norms = np.linalg.norm(self.embedded_documents, axis=1, keepdims=True)
        doc_norms[doc_norms < 1e-12] = 1.0
        normalized_docs = self.embedded_documents / doc_norms
        scores = normalized_docs @ query_embedding
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        return [self.documents[idx] for idx in top_indices]


class RAGChatbot:

    def __init__(self):

        print("Loading embeddings...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print("Loading documents...")
        self.documents = self.load_documents()
        self.retriever = SimpleRetriever(self.documents, self.embeddings)

        print("Loading FLAN-T5...")

        self.tokenizer, self.model = load_seq2seq_model(LLM_MODEL)

        print("Ready!")

    # ===================================================
    # LOAD CSV
    # ===================================================

    def load_documents(self):

        df = pd.read_csv(
            CSV_PATH,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip"
        )

        documents = []

        for _, row in df.iterrows():

            text = ""

            for col in df.columns:

                value = str(row[col])

                text += f"{col}: {value}\n"

            documents.append(
                Document(
                    page_content=text
                )
            )

        return documents

    # ===================================================
    # VECTOR STORE
    # ===================================================

    def create_vectorstore(self):
        return None

    # ===================================================
    # ASK
    # ===================================================

    def ask(self, question):

        docs = self.retriever.invoke(question)

        if not docs:
            return "I couldn't find any relevant information in the dataset."

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
    You are an experienced software engineer and Stack Overflow expert.

    Use ONLY the information available in the retrieved context.

    Retrieved Context:
    {context}

    User Question:
    {question}

    Instructions:
    - Answer in 5-8 complete sentences.
    - Explain the concept clearly.
    - Provide the correct solution.
    - Mention best practices.
    - If the retrieved context contains code, include it.
    - Do not invent information outside the retrieved context.
    - If the context is insufficient, say:
    "I couldn't find enough information in the dataset."

    Answer:
    """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.3,
            do_sample=True
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()