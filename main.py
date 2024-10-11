import streamlit as st
import os
from langchain_groq import ChatGroq
from src.chains import create_vector_embedding, load_document, load_url
from src.chains import create_chain, clear_session_state_documents_vectors, create_vector_db

def main():
    st.title("RAG Document Q&A")
    st.write("Upload pdf/url and ask Q&A - Groq and Llama3")

    # from dotenv import load_dotenv
    # load_dotenv()
    # ## load the GROQ API Key
    # api_key=os.getenv("GROQ_API_KEY")

    ## Input the Groq API Key
    st.subheader("1. Enter Groq API Key")

    api_key=st.text_input("Enter your Groq API key:",type="password",placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxx")

    ## Check if groq api key is provided
    try:

        if api_key:
            llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192")

            # Input Job Listing
            st.subheader("2. Upload PDF or enter URL")
            upload_type = st.radio("Select input type", ("PDF", "URL"))

            if upload_type == "PDF":
                # clear_session_state_documents_vectors()
        
                uploaded_files=st.file_uploader("Choose single/multiple .pdf files",type="pdf",accept_multiple_files=True)
                # if uploaded_files:
                if st.button("Fetch content"):
                    # with st.spinner("Loading URL content..."):                
                    clear_session_state_documents_vectors(st)
                    st.session_state.documents=load_document(uploaded_files=uploaded_files)
                    create_vector_db(st)

            elif upload_type == "URL":
                # clear_documents_vectors_session_state()
                upload_url = st.text_input("Enter URL", "")
                if st.button("Fetch content"):
                    # with st.spinner("Loading URL content..."):
                    clear_session_state_documents_vectors(st)
                    st.session_state.documents=load_url(upload_url)
                    # print("st.session_state4:--------------\n:",st.session_state)
                    create_vector_db(st)

            if "vectors" in st.session_state:
                st.subheader("3. Ask questions based on uploaded document")

                user_prompt=st.text_input("Enter your query from the uploaded document")
                submit=st.button("Submit")    

                if submit:
                    with st.spinner("AI thinking..."):
                        create_chain(user_prompt,llm)

            
    except Exception as e:
        st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    main()