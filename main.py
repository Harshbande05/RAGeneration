from fastapi import FastAPI, UploadFile, File, HTTPException
from inference.infer import response
from train.train import train_document
from typing import List
from pydantic import BaseModel
import os
import shutil

app = FastAPI()

DATA_DIR = "././data"


class InferPrompt(BaseModel):
    question: str


@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "Connection live!"}


@app.post("/train")
async def train(files: List[UploadFile] = File(...)):
    """
    Upload files and trigger training.
    """
    try:
        # Clear the data directory if it exists
        if os.path.isdir(DATA_DIR):
            shutil.rmtree(DATA_DIR, ignore_errors=True)

        # Create the upload directory
        os.makedirs(DATA_DIR, exist_ok=True)

        # Save uploaded files
        for file in files:
            file_path = os.path.join(DATA_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Start the training process
        result = train_document()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")


@app.post("/infer")
async def infer(question: InferPrompt):
    """
    Handle inference for a given question.
    """
    try:
        # Call the inference function with the provided question
        result = response(question.question)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during inference: {str(e)}")
