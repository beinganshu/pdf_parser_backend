from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from main import extract_text_from_pdf, clean_and_chunk_text, index_text_chunks, query_gemini_ai, save_model, load_model
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
active_sessions = {}
# @app.get("/")
# def home():
#     return {"message":"Hello World!"}
@app.post("/upload/")
def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save file
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text and index the PDF
        raw_text = extract_text_from_pdf(file_path, use_ocr=True)
        text_chunks = clean_and_chunk_text(raw_text)
        index, embeddings, model, text_chunks = index_text_chunks(text_chunks)

        # Save the model
        model_file_path = f"models/{file.filename}.pkl"
        save_model(model_file_path, index, model, text_chunks)

        # Add to active sessions
        active_sessions[file.filename] = {
            "uploaded": True,
            "model_file_path": model_file_path
        }

        return JSONResponse(content={"upload-message": "File uploaded and indexed successfully"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})


@app.post("/query/")
def query_pdf(query: str = Form(...)):
    # Check if there's an active session
    active_file = None
    for filename, session in active_sessions.items():
        if session.get("uploaded"):
            active_file = filename
            break
    
    if not active_file:
        return JSONResponse(content={"error": "No file uploaded or session has ended"})
    
    # Handle the "quit" query to stop the session
    if query.lower() == "quit":
        model_file_path = active_sessions[active_file]["model_file_path"]
        
        # Delete the model after quitting the session
        if os.path.exists(model_file_path):
            os.remove(model_file_path)
        
        # Remove the session from active_sessions
        active_sessions[active_file]["uploaded"] = False
        
        return JSONResponse(content={"message": "Query session ended and model deleted."})
    
    # Process the query with the current active file
    try:
        model_file_path = active_sessions[active_file]["model_file_path"]
        index, model, text_chunks = load_model(model_file_path)
        response = query_gemini_ai(query, index, model, text_chunks)
        return JSONResponse(content={"response": response})
    
    except FileNotFoundError:
        return JSONResponse(content={"error": "Model file not found for this session"})
