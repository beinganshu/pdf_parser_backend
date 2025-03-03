import fitz
import pytesseract
from pdf2image import convert_from_path
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import shutil
import os

def extract_text_from_pdf(pdf_path, use_ocr=False, ocr_lang='eng'):
    doc = fitz.open(pdf_path)
    extracted_text = []
    
    for page in doc:
        text = page.get_text("text")
        if not text.strip() and use_ocr:
            images = convert_from_path(pdf_path, first_page=page.number + 1, last_page=page.number + 1)
            for img in images:
                text = pytesseract.image_to_string(img, lang=ocr_lang)
        extracted_text.append(text.strip())
    
    doc.close()
    return extracted_text

def clean_and_chunk_text(text_list, chunk_size=500):
    full_text = "\n".join(text_list)
    full_text = re.sub(r'\s+', ' ', full_text)
    words = full_text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def index_text_chunks(text_chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = np.array(model.encode(text_chunks))
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, model, text_chunks

def query_gemini_ai(query, index, model, text_chunks, top_k=5):
    query_embedding = np.array([model.encode(query)])
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n".join(relevant_chunks)
    
    genai.configure(api_key="AIzaSyBSGr1p4lbcwuf01K2Cb3mxAWewOf9I5z8")
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(f"Based on the following information, answer the question: {query}\n\n{context}")
    return response.text

import pickle

def save_model(file_path, index, model, text_chunks):
    """ Saves the FAISS index and text chunks. """
    with open(file_path, 'wb') as f:
        pickle.dump((index, model, text_chunks), f)

def load_model(file_path):
    """ Loads the FAISS index and text chunks. """
    with open(file_path, 'rb') as f:
        return pickle.load(f)