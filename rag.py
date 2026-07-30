# import os
# import pandas as pd

# from config import *

# from langchain_core.documents import Document

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.vectorstores import FAISS

# from transformers import pipeline

# from langchain_huggingface import HuggingFacePipeline

# from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.output_parsers import StrOutputParser

# from langchain_core.runnables import RunnablePassthrough


# class RAGChatbot:

#     def __init__(self):

#         self.embeddings = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL
#         )

#         self.vectorstore = self.load_or_create_vectorstore()

#         self.retriever = self.vectorstore.as_retriever(
#                 search_type="similarity_score_threshold",
#                 search_kwargs={
#                     "k": TOP_K,
#                     "score_threshold": 0.55
#                 }
#             )

#         pipe = pipeline(
#             task="text2text-generation",
#             model=LLM_MODEL,
#             max_new_tokens=200,
#             temperature=0
#         )

#         self.llm = HuggingFacePipeline(pipeline=pipe)

#         self.prompt = ChatPromptTemplate.from_template(
#             """
# You are an AI assistant.

# You are a professional Stack Overflow assistant.

# Use ONLY the retrieved answers below.

# If multiple retrieved answers are relevant,
# combine them into one clear answer.

# If the answer is missing,
# reply exactly:

# I don't know based on the provided documents.

# Retrieved Answers:
# {context}

# Question:
# {question}

# Write a complete answer in 4-8 sentences.

# If the answer is unavailable in the context,
# reply with:

# "I don't know based on the provided documents."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#         )

#         self.chain = (
#             {
#                 "context": self.retriever | self.format_docs,
#                 "question": RunnablePassthrough()
#             }
#             | self.prompt
#             | self.llm
#             | StrOutputParser()
#         )

#     def load_documents(self):

#         df = pd.read_csv(CSV_PATH)

#         documents = []

#         for _, row in df.iterrows():

#             question = str(row["Question"])

#             answer = str(row["Answer"])

#             text = f"Question: {question}\nAnswer: {answer}"

#             documents.append(
#                 Document(
#                     page_content=text,
#                     metadata={
#                         "question": question
#                     }
#                 )
#             )

#         return documents

#     def split_documents(self, documents):

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )

#         return splitter.split_documents(documents)

#     def load_or_create_vectorstore(self):

#         if os.path.exists(VECTOR_DB):

#             return FAISS.load_local(
#                 VECTOR_DB,
#                 self.embeddings,
#                 allow_dangerous_deserialization=True
#             )

#         docs = self.load_documents()

#         chunks = self.split_documents(docs)

#         db = FAISS.from_documents(
#             chunks,
#             self.embeddings
#         )

#         db.save_local(VECTOR_DB)

#         return db

#     def format_docs(self, docs):

#         return "\n\n".join(doc.page_content for doc in docs)

#     def ask(self, question):

#         docs = self.vectorstore.similarity_search_with_score(question, k=3)

#         if not docs:
#             return "I don't know based on the provided documents."

#         best_doc, score = docs[0]

#         if score < 0.35:
#             # Very close match → return stored answer directly
#             return best_doc.page_content.replace("Question:", "").replace("Answer:", "").strip()

#         context = "\n\n".join(doc.page_content for doc, _ in docs)

#         prompt = f"""
#     Use ONLY the retrieved answers.

#     Retrieved Answers:
#     {context}

#     Question:
#     {question}

#     Generate one concise Stack Overflow style answer.
#     """

#         return self.llm.invoke(prompt)



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
            model_name="sentence-transformers/all-MiniLM-L6-v2"
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
            task="text2text-generation",
            model=model_name,
            max_length=512
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