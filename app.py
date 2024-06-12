import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy import create_engine
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import re
import httpx
import logging
import os

# huggingface token
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
api_token = os.getenv('API_TOKEN')

engine = create_engine(DATABASE_URL)

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_connection():
    try:
        response = httpx.get(SUPABASE_URL)
        st.write("Supabase URL Response Status Code:", response.status_code)
    except Exception as e:
        st.error(f"Failed to connect to Supabase URL: {e}")

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        text = None  # Set text to None in case of error
    return text

def get_text_chunks(text):
    max_chunk_size = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=10,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_embeddings(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return embeddings

def find_top_chunks(user_query, content, content_embeddings, top_n=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([user_query], convert_to_tensor=True)
    similarities = np.dot(content_embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_chunks = [content[idx] for idx in top_indices]
    return top_chunks

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,@-]', '', text)
    return text.strip()

def extract_chunks(texts):
    cleaned_chunks = [clean_text(text) for text in texts]
    return cleaned_chunks

def handle_userinput(user_question):
    id_value = st.session_state.get('id')  # Safely access session state with .get() method
    # st.write("ID value from session state:", id_value)
    
    if id_value is None:
        st.error("User ID is not set. Please set your user ID first.")
        return

    response = supabase_client.table('pdfs').select('embeddings', 'content').eq('id', id_value).execute()
    # st.write("Response from supabase:", response.data)
    
    if not response.data:  # Check if response data is empty
        st.error("No data found for the provided user ID.")
        return

    data = response.data[0]
    content = data.get('content')  # Safely access 'content' key
    content_embeddings = np.array(data.get('embeddings', []))  # Safely access 'embeddings' key

    if isinstance(content, str):
        try:
            content = eval(content)
        except Exception as e:
            st.error(f"Content is not in the expected format: {e}")
            return

    best_chunk = find_top_chunks(user_question, content, content_embeddings)
    # st.write("Best chunk found:", best_chunk)
    
    best_chunk = extract_chunks(best_chunk)

    input_text = f"You are an AI language model and will answer the query based on the best chunk provided. Query: {user_question} Best chunk: {best_chunk}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": input_text, "parameters": {"max_length": 512, "temperature": 0.7, "repetition_penalty": 1.2}}

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json=payload
    )
    # st.write("Response from HuggingFace API:", response.status_code, response.content)
    
    if response.status_code == 200:
        response_data = response.json()
        if isinstance(response_data, list) and 'generated_text' in response_data[0]:
            response_text = response_data[0]['generated_text']
        else:
            response_text = "Sorry, I couldn't generate a response. Please try again."
    else:
        response_text = f"Error: {response.status_code}. {response.content.decode('utf-8')}"

    response_text = ' '.join(dict.fromkeys(response_text.split()))

    st.session_state.chat_history = [
        {"role": "user", "content": user_question},
        {"role": "bot", "content": response_text}
    ]

    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

def fetch_user_data(id_value):
    response = supabase_client.table('pdfs').select('id').eq('id', id_value).limit(1).execute()
    # st.write("Fetch user data response:", response.data)
    
    existing_data = response.data[0]
    if existing_data['id']:
        return True
    else:
        st.error(f"No data found for ID, try again!: {id_value}")
        return False

def main():
    logging.debug("Starting the main function")
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "old_id_value" not in st.session_state:
        st.session_state.old_id_value = None
    if "new_id_value" not in st.session_state:
        st.session_state.new_id_value = None
    if "old_user_pdf_docs" not in st.session_state:
        st.session_state.old_user_pdf_docs = None
    if "new_user_pdf_docs" not in st.session_state:
        st.session_state.new_user_pdf_docs = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    st.header("Chat with multiple PDFs :books:")

    st.subheader("if this is your first time, create user id to save your documents")

    new_id_value = st.text_input("Enter your ID", value="")
    logging.debug(f"New ID value entered: {new_id_value}")

    # Unique keys for each file uploader
    key_old_user = "file_uploader_old_user"
    key_new_user = "file_uploader_new_user"

    with st.sidebar:
        logging.debug("Inside sidebar")
        # for old users
        st.subheader("if you want to continue with your old user id, enter the same id")
        id_value = st.text_input("Enter your old ID", value="")
        logging.debug(f"Old ID value entered: {id_value}")
        
        if st.button("Continue with old ID"):
            logging.debug("Continue with old ID button pressed")
            if fetch_user_data(id_value):
                st.session_state.old_id_value = id_value  # Save to session state
                st.write("Welcome back!")
                st.subheader("if you want to add new documents, upload them below")
                if st.session_state.old_id_value is not None:
                    # Display file uploader only if "Upload PDFs" button is clicked
                    old_user_pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, key="key_old_user")
                    logging.debug(f"Old user PDF docs: {old_user_pdf_docs}")  # Debugging
                    if old_user_pdf_docs is None or len(old_user_pdf_docs) == 0:
                        st.write("No PDFs uploaded.")
                    else:
                        st.session_state.old_user_pdf_docs = old_user_pdf_docs  # Save to session state
                        st.write("PDFs uploaded successfully.")


        # for new users
        st.subheader("If You are New, Enter Your documents")
        new_user_pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, key=key_new_user)
        st.session_state.new_user_pdf_docs = new_user_pdf_docs  # Save to session state

        if st.button("Process"):
            logging.debug("Process button pressed")
            old_id_value = st.session_state.get('old_id_value')
            old_user_pdf_docs = st.session_state.get('old_user_pdf_docs')

            logging.debug(f"old_user_pdf_docs: {old_user_pdf_docs}") # Debugging
            
            if ((old_id_value and old_user_pdf_docs) or (new_id_value and new_user_pdf_docs)):
                with st.spinner("Processing"):
                    try:
                        if old_id_value and old_user_pdf_docs:
                            try:
                                # Fetch existing data associated with old_id_value
                                response = supabase_client.table('pdfs').select('content', 'embeddings').eq('id', old_id_value).execute()
                                
                                if response.data:
                                    logging.debug("Response from Supabase for old user:")  # Debugging
                                    logging.debug(response.data)
                                    
                                    existing_data = response.data[0]
                                    existing_content = existing_data.get('content', [])
                                    existing_embeddings = np.array(existing_data.get('embeddings', []))
                                    
                                    # Process new PDF documents
                                    new_raw_text = get_pdf_text(old_user_pdf_docs)
                                    new_text_chunks = get_text_chunks(new_raw_text)
                                    new_embeddings = get_embeddings(new_text_chunks)
                                    
                                    # Combine existing and new data
                                    combined_text_chunks = existing_content + new_text_chunks
                                    combined_embeddings = np.concatenate([existing_embeddings, new_embeddings], axis=0)
                                    combined_embedding_list = combined_embeddings.tolist()
                                    
                                    # Prepare data for insertion
                                    data = {'id': old_id_value, 'content': combined_text_chunks, 'embeddings': combined_embedding_list}
                                    
                                    # Perform database operations
                                    supabase_client.table('pdfs').delete().eq('id', old_id_value).execute()
                                    supabase_client.table('pdfs').insert(data, on_conflict=('id', 'update')).execute()
                                    
                                    # Update session state flags
                                    st.session_state.id = old_id_value
                                    st.session_state.pdf_processed = True
                                else:
                                    st.write("No data found for the provided ID.")
                                    
                            except Exception as e:
                                st.error(f"An error occurred: {e}")
                        elif new_id_value and new_user_pdf_docs:
                            raw_text = get_pdf_text(new_user_pdf_docs)
                            logging.debug(f"New raw text for new user: {raw_text}")
                            
                            text_chunks = get_text_chunks(raw_text)
                            embeddings = get_embeddings(text_chunks)
                            embedding_list = embeddings.tolist()
                            st.session_state.id = new_id_value
                            data = {'id': new_id_value, 'content': text_chunks, 'embeddings': embedding_list}
                            supabase_client.table('pdfs').insert(data).execute()
                            st.session_state.pdf_processed = True
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
            if old_id_value and not old_user_pdf_docs:
                with st.spinner("Processing"):
                    st.session_state.id = old_id_value
                    st.session_state.pdf_processed = True
                    st.write("Only old ID provided, no new documents")

            st.success("Processing complete!")
            st.write("You can now ask a question to the chatbot.")

    if st.session_state.pdf_processed:
        user_question = st.text_input("Ask a question to the chatbot")
        if st.button("Get Answer"):
            if user_question:
                handle_userinput(user_question)
            else:
                st.error("Please enter a question to get an answer.")
    else:
        st.warning("Please upload your PDFs")

if __name__ == "__main__":
    main()
