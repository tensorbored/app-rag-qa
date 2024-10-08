from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
# from dotenv import load_dotenv
# load_dotenv()

def load_document(uploaded_files):
    ## Process uploaded PDF's
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./tmp/temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    return documents

def create_vector_embedding(documents):

        # st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
        # st.session_state.loader=PyPDFLoader("CV_Prateek.pdf") ## Data Ingestion step
        # print("st.session_state.loader",st.session_state.loader)
        # st.session_state.documents=st.session_state.loader.load() ## Document Loading
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_documents=text_splitter.split_documents(documents)
        embeddings=OllamaEmbeddings(model="all-minilm")
        vectors=Chroma.from_documents(final_documents,embeddings)
        # st.session_state.retriever = st.session_state.vectors.as_retriever()
        return vectors


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