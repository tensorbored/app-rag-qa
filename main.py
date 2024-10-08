import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from chains import create_vector_embedding,load_document,prompt

def main():
    st.title("RAG Document Q&A")
    st.subheader("Upload pdf's and ask Q&A with their content - Groq and Llama3")

    # from dotenv import load_dotenv
    # load_dotenv()
    # ## load the GROQ API Key
    # api_key=os.getenv("GROQ_API_KEY")

    ## Input the Groq API Key
    api_key=st.text_input("Enter your Groq API key:",type="password")

    ## Check if groq api key is provided
    if api_key:
        llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192")
        uploaded_files=st.file_uploader("Choose single/multuple .pdf files",type="pdf",accept_multiple_files=True)
        try:
            documents=load_document(uploaded_files=uploaded_files)

            if st.button("Document Embedding"):
                with st.spinner("Creating Vector Database..."):
                    if "vectors" in st.session_state:
                        st.write("Vector Database is already created")
                    if "vectors" not in st.session_state:
                        st.session_state.vectors=create_vector_embedding(documents=documents)
                        st.write("Vector Database is ready")

            if "vectors" in st.session_state:
                user_prompt=st.text_input("Enter your query from the uploaded document")
                submit=st.button("Submit")    

                if submit:
                    # document_chain=create_stuff_documents_chain(llm,prompt)
                    document_chain = prompt | llm | StrOutputParser()

                    retriever=st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
                    retrieval_chain=create_retrieval_chain(retriever,document_chain)

                    response=retrieval_chain.invoke({'input':user_prompt})
                    st.write(response['answer'])

                    # With a streamlit expander
                    with st.expander("Document similarity Search"):
                        for i,doc in enumerate(response['context']):
                            st.write(doc.page_content)
                            st.write('------------------------')
        
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    main()