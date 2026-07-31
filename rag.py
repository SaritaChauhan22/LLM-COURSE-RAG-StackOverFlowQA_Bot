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
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

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


class RAGChatbot:

    def __init__(self):

        print("Loading embeddings...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print("Loading Vector Store...")

        self.vectorstore = self.create_vectorstore()

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )

        print("Loading FLAN-T5...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL
        )

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

        if os.path.exists(os.path.join(VECTOR_DB, "index.faiss")):

            print("Loading existing FAISS database...")

            return FAISS.load_local(
                VECTOR_DB,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        print("Creating FAISS database...")

        documents = self.load_documents()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(documents)

        db = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        db.save_local(VECTOR_DB)

        return db

    # ===================================================
    # ASK
    # ===================================================

    def ask(self, question):

        docs = self.retriever.invoke(question)

        if len(docs) == 0:

            return "I couldn't find any relevant information."

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = f"""
You are a helpful StackOverflow assistant.

Answer ONLY using the context below.

If the answer is not present in the context,
reply:

I couldn't find the answer in the dataset.

Context:

{context}

Question:

{question}

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
            max_new_tokens=150,
            do_sample=False
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()