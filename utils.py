from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings

warnings.filterwarnings("ignore")

import os

from dotenv import load_dotenv
load_dotenv() 

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
embedding_encoder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vectordb_file_path = "vector_store"
input_file_path = "resources/codebasics_faqs.csv"



def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path=input_file_path, source_column="prompt")
    data = loader.load()
    
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=embedding_encoder)

    vectordb.save_local(vectordb_file_path)
    
    
def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embedding_encoder, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke("Do you have javascript course?"))