import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# import chromadb

# from dotenv import load_dotenv
# load_dotenv()

def load_document(uploaded_files):
    ## Process uploaded PDF's
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./assets/temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            # file_name=uploaded_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    return documents

def load_url(upload_url):
    loader=WebBaseLoader(upload_url)
    documents=loader.load()
    return documents

def create_vector_embedding(documents):

        # st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
        # st.session_state.loader=PyPDFLoader("CV_Prateek.pdf") ## Data Ingestion step
        # print("st.session_state.loader",st.session_state.loader)
        # st.session_state.documents=st.session_state.loader.load() ## Document Loading
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_documents=text_splitter.split_documents(documents)
        embeddings=OllamaEmbeddings(model="all-minilm")
        # vector_store.delete(ids=uuids[-1])
        # the below deletes all the chunks from the doc1 file
        # vectors=Chroma.from_documents(final_documents,embeddings) #,persist_directory='./repository/db')
        vectors=InMemoryVectorStore.from_documents(final_documents,embeddings) #,persist_directory='./repository/db')
        # print("vectors:-----------------------------------------------------",vectors)
        # st.session_state.retriever = st.session_state.vectors.as_retriever()
        return vectors

def clear_session_state_documents_vectors(st):
    if "documents" in st.session_state:
        del st.session_state["documents"]
    if "vectors" in st.session_state:
        # https://github.com/langchain-ai/langchain/discussions/9495#discussioncomment-10503820
        st.session_state.vectors.reset_collection()
        del st.session_state["vectors"]

def create_vector_db(st):
    if "documents" in st.session_state:
        with st.spinner("Creating Vector Database..."):
            if "vectors" in st.session_state:
                st.write("Vector Database is already created")
            elif "vectors" not in st.session_state:
                st.session_state.vectors=create_vector_embedding(documents=st.session_state.documents)
                st.write("Vector Database is ready")

def create_chain(user_prompt,llm):

    prompt=ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate respone based on the question
            <context>
            {context}
            <context>
            Question:{input}
            """
        )
    # document_chain=create_stuff_documents_chain(llm,prompt)
    document_chain = prompt | llm | StrOutputParser()
    retriever=st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response=retrieval_chain.invoke({'input':user_prompt})
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document similarity Search: Top 3"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')


