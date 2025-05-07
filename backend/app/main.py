"""
This module defines a FastAPI application for the backend of the Financial Security System.
Running as a script will start a development uvicorn server serving the app.
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    """
    Handles GET requests to the root endpoint.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {
        "message": "Welcome to backend FastAPI server for Financial Security System!"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
