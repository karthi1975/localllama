import os
import streamlit as st
import fitz  # PyMuPDF for reading PDFs
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from uuid import uuid4
import numpy as np
from better_profanity import profanity  # For filtering inappropriate content
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import torch
import re

# Initialize profanity filtering
profanity.load_censor_words()

# Function to read and preprocess documents from a PDF file
def read_documents_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    documents = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():  # Ensure that the page has text
            documents.append(Document(page_content=text))
    return documents

# Custom Docstore class to manage documents
class SimpleDocstore:
    def __init__(self):
        self.docs = {}

    def add_documents(self, documents, ids):
        for doc, doc_id in zip(documents, ids):
            self.docs[doc_id] = doc

    def get_document(self, doc_id):
        return self.docs.get(doc_id, None)

# Initialize FAISS vector store
def initialize_faiss_store(documents, embeddings_model):
    # Embed documents
    texts = [doc.page_content for doc in documents]
    embedded_texts = embeddings_model.embed_documents(texts)
    embedded_texts = np.vstack(embedded_texts)  # Stack embeddings into a 2D array
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embedded_texts.shape[1])

    # Initialize docstore (InMemoryDocstore is part of Langchain)
    docstore = InMemoryDocstore()

    # Create a mapping of document IDs to FAISS indices
    index_to_docstore_id = {i: str(uuid4()) for i in range(len(documents))}

    # Add documents to the docstore
    for i, doc_id in index_to_docstore_id.items():
        docstore._dict[doc_id] = documents[i]  # Proper way to add documents to InMemoryDocstore

    # Initialize FAISS vector store
    faiss_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,  # Use Langchain's InMemoryDocstore
        index_to_docstore_id=index_to_docstore_id
    )

    # Add embeddings to FAISS index
    faiss_store.add_documents(documents)

    return faiss_store

# FAISS search function
def search_faiss(query, faiss_store):
    docs_and_scores = faiss_store.similarity_search_with_score(query)
    
    # If no documents are found, return an error message
    if not docs_and_scores:
        return "No relevant documents found."
    
    return docs_and_scores

# Guardrail function using better_profanity
def apply_guardrails(response_content):
    # Check for profanity and censor it
    if profanity.contains_profanity(response_content):
        response_content = profanity.censor(response_content)
    
    # Additional guardrails for length or other content checks can be added here
    if len(response_content) < 50:
        return "The response seems too short. Please try rephrasing your question."
    
    return response_content


# Function to determine if a line is a complete statement (not a question or incomplete)
def is_complete_statement(line):
    # Check if the line ends with a period, exclamation mark, or appropriate punctuation, 
    # but not a question mark
    return bool(re.match(r".*[\.\!\]]$", line.strip())) and not line.strip().endswith('?')

# Function to extract only the answer part and remove any incomplete or redundant content
def extract_and_format_answer(response_content, max_points=2):
    # Focus only on the "Answer:" portion and clean it up
    if "Answer:" in response_content:
        answer_content = response_content.split("Answer:")[-1].strip()
    else:
        answer_content = response_content.strip()

    # Remove any unnecessary characters like "<< >>"
    formatted_answer = answer_content.replace("<<", "").replace(">>", "").strip()

    # Split the answer into lines and filter to get the required points
    formatted_answer_lines = formatted_answer.split("\n")
    cleaned_lines = []
    count = 1

    for line in formatted_answer_lines:
        # Check if the line is a complete statement and not a question
        if line.strip() and is_complete_statement(line.strip()):
            # Ensure proper formatting of points without repeating the number
            cleaned_lines.append(f"{count}. {line.strip().lstrip('0123456789. ')}")
            count += 1
            if count > max_points:
                break  # Stop once the desired number of points is reached

    # Remove any hanging or incomplete sentences
    final_answer_lines = [line for line in cleaned_lines if not re.match(r'^\d+\.\s*$', line.strip())]

    # Return cleaned response or indicate no content found
    if not final_answer_lines:
        return "No relevant content extracted. Please refine your question."

    final_answer = "\n\n".join(final_answer_lines)
    
    return final_answer



# Initialize the pipeline with LLaMA for RAG
def initialize_pipeline(documents, model_choice="LLaMA"):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_store = initialize_faiss_store(documents, embeddings_model)
    
    # Load LLaMA model for text generation
    model_path = "./Llama-3.2-3B"  # Ensure your local model is at this path
   
    # Ensure that the local model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"The model path {model_path} does not exist.")

    llama_pipeline = pipeline(
        "text-generation", 
        model=model_path, 
        torch_dtype=torch.float16,  # Use float16 for optimized performance
        device_map="auto",  # Automatically map to GPU if available
        max_new_tokens=150,  # Limit token generation to avoid repetition
        temperature=0.5,  # Adjust temperature to add variety
        top_k=50,  # Use top-k sampling to avoid repetition
        top_p=0.9  # Use top-p sampling for diverse responses
    )
    
    return llama_pipeline, faiss_store

# Streamlit Chat Interface
st.title("Converse With you Data")

# Provide download link for the default PDF
default_pdf_path = os.path.join(os.getcwd(), "principals_ethic_ai.pdf")
if os.path.exists(default_pdf_path):
    with open(default_pdf_path, "rb") as f:
        st.download_button(
            label="Download default PDF (principals_ethic_ai.pdf)",
            data=f,
            file_name="principals_ethic_ai.pdf",
            mime="application/pdf"
        )
else:
    st.error("Default PDF not found. Please upload a PDF file.")

# Initialize session state for messages if not already done
if 'messages' not in st.session_state:
    st.session_state.messages = {"LLaMA": []}
if 'current_model' not in st.session_state:
    st.session_state.current_model = "LLaMA"

# Automatically load the default PDF if available
if os.path.exists(default_pdf_path):
    with open(default_pdf_path, "rb") as default_pdf:
        documents = read_documents_from_pdf(default_pdf)
    st.success("Default PDF loaded and processed successfully!")
else:
    documents = []

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a different PDF file", type=["pdf"])

if uploaded_file:
    documents = read_documents_from_pdf(uploaded_file)
    st.success("PDF uploaded and processed successfully!")

# Full context remains the same, replace the extract_and_format_answer function with the revised version.
# Ensure that you use the new function during response handling:

# ... (rest of your code)

# Use the updated extract_and_format_answer during the response handling
if documents:
    llama_pipeline, faiss_store = initialize_pipeline(documents)

    # Chat interface
    st.write("## Chat")
    for message in st.session_state.messages[st.session_state.current_model]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages[st.session_state.current_model].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Search for relevant context using FAISS
                search_results = search_faiss(prompt, faiss_store)
                context = " ".join([doc.page_content for doc, _ in search_results[:5]])

                # Construct the full prompt for the LLaMA model, focusing only on the answer
                full_prompt = f"""
                Based on the document, answer the following question as accurately as possible:

                Question: {prompt}

                Answer:
                """
                
                # Generate response using the LLaMA model
                result = llama_pipeline(full_prompt)
                response_content = result[0]['generated_text']

                # Apply guardrails to the generated response
                response_content = apply_guardrails(response_content)

                # Extract and format the answer to ensure quality output without repetitions
                formatted_answer = extract_and_format_answer(response_content, max_points=2)

                # Display the formatted answer using Streamlit markdown
                st.markdown(formatted_answer)
                st.session_state.messages[st.session_state.current_model].append({"role": "assistant", "content": formatted_answer})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please upload a PDF file to start the conversation.")

st.markdown("---")
st.write("This application utilizes the LLaMA model to provide answers based on content from uploaded PDF documents.")