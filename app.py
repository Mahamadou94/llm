# app.py
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from typing import List
from generation import ReviewProcessor  # Adjust this import based on your actual file structure
import os
import uvicorn

app = FastAPI(title="Document Analysis API")

processor = None  # Global processor instance

@app.on_event("startup")
async def startup_event():
    global processor
    processor = ReviewProcessor("config.ini")
    await processor.process()

@app.get("/files", response_model=List[str])
def list_files():
    """ List all analysis files available for download. """
    files = [f for f in os.listdir('.') if f.startswith('analysis_chunk')]
    return files

@app.get("/download/{filename}", response_class=FileResponse)
def download_file(filename: str):
    """ Endpoint to download a specific analysis file by filename """
    if filename in os.listdir('.'):
        return FileResponse(path=filename, filename=filename, media_type='application/octet-stream')
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    # Configuration to run the server with Uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    
