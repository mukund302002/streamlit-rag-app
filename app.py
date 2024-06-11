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
# huggingface token
api_token="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"

DATABASE_URL = 'postgresql+psycopg2://postgres.hjjjkgzdxwstwuqjspje:eDCsslfp0QmY0Ijk@aws-0-ap-south-1.pooler.supabase.com:6543/postgres'


engine = create_engine(DATABASE_URL)
# Ensure to set your environment variables or replace these values accordingly
SUPABASE_URL = "https://hjjjkgzdxwstwuqjspje.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhqamprZ3pkeHdzdHd1cWpzcGplIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTc1NzA3OTcsImV4cCI6MjAzMzE0Njc5N30.sWTlOvOS7PcTBmwgdLOB6hxli4ohUM7IZfMNTlL1gyE"



supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_connection():
    try:
        response = httpx.get(SUPABASE_URL)
        # st.write("Supabase URL Response Status Code:", response.status_code)
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
    if id_value is None:
        st.error("User ID is not set. Please set your user ID first.")
        return

    response = supabase_client.table('pdfs').select('embeddings', 'content').eq('id', id_value).execute()
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
    existing_data = response.data[0]
    if existing_data['id']:
        return True
    else:
        st.error(f"No data found for ID, try again!: {id_value}")
        return False




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None
    if "id" not in st.session_state:
        st.session_state.id = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    st.header("Chat with multiple PDFs :books:")

    st.subheader("Welcome to the AI-powered document chatbot!")
    st.subheader("if this is your first time, create user id to save your documents")

    new_id_value = st.text_input("Enter your ID", value="")
    # st.session_state.id = id_value

    # Initialize variables for old user ID and PDF docs
    old_id_value = None
    old_user_pdf_docs = None
    # Unique keys for each file uploader
    key_old_user = "file_uploader_old_user"
    key_new_user = "file_uploader_new_user"
    

    with st.sidebar:
        # for old users
        st.subheader("if you want to continue with your old user id, enter the same id")
        id_value = st.text_input("Enter your old ID", value="")
        if st.button("Continue with old ID"):
            if(fetch_user_data(id_value)):
                old_id_value = id_value
                st.write("Welcome back!")
                st.subheader("if you want to add new documents, upload them below")
                old_user_pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, key=key_old_user)
            
        # for new users
        st.subheader("If You are New, Enter Your documents")
        new_user_pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, key=key_new_user)


        if st.button("Process"):
            if ((old_id_value and old_user_pdf_docs) or (new_id_value and new_user_pdf_docs)):
                with st.spinner("Processing"):
                    try:
                        if old_id_value and old_user_pdf_docs:
                            try:
                                # Fetch existing data associated with old_id_value
                                response = supabase_client.table('pdfs').select('content', 'embeddings').eq('id', old_id_value).execute()
                                existing_data = response.data[0]
                                existing_content = existing_data['content'] if 'content' in existing_data else []
                                existing_embeddings = np.array(existing_data['embeddings']) if 'embeddings' in existing_data else np.array([])
                                
                                # Process new PDF documents
                                new_raw_text = get_pdf_text(old_user_pdf_docs)
                                new_text_chunks = get_text_chunks(new_raw_text)
                                new_embeddings = get_embeddings(new_text_chunks)
                                
                                # Combine existing and new data
                                combined_text_chunks = existing_content + new_text_chunks
                                combined_embeddings = np.concatenate([existing_embeddings, new_embeddings], axis=0)
                                combined_embedding_list = combined_embeddings.tolist()
                                
                                # Insert combined data back into the table
                                data = {'id': old_id_value, 'content': combined_text_chunks, 'embeddings': combined_embedding_list}
                                supabase_client.table('pdfs').insert(data, on_conflict=('id', 'update')).execute()
                                
                                # Update session state flags
                                st.session_state.id = old_id_value
                                st.session_state.pdf_processed = True
                            except Exception as e:
                                st.error(f"An error occurred: {e}")

                        elif new_id_value and new_user_pdf_docs:
                            raw_text = get_pdf_text(new_user_pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            embeddings = get_embeddings(text_chunks)
                            embedding_list = embeddings.tolist()
                            st.session_state.id = new_id_value
                            data = {'id': new_id_value, 'content': text_chunks, 'embeddings': embedding_list}
                            supabase_client.table('pdfs').insert(data).execute()
                            st.session_state.pdf_processed = True
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
            elif old_id_value and not old_user_pdf_docs:
                with st.spinner("Processing"):
                    st.session_state.id = old_id_value
                    st.session_state.pdf_processed = True
                    st.write("mae chla hun bhai")
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
        st.warning("Processing is not complete. First upload your PDFs")
    

if __name__ == "__main__":
    main()
