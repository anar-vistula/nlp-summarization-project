"""
Entry point to run the Text Summarization API server.
Usage: python run.py
"""

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from dotenv import load_dotenv
load_dotenv()  # Load .env file (API keys)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
