from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import gradio as gr
import uvicorn
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ====================================================================================================

gradio_app123 = gr.Interface(lambda aaaaa: "Hello, " + aaaaa + "!", "textbox", "textbox")

# ====================================================================================================

# ENDPOINT: /root
@app.get("/")
def endpoint_root():
    return {"response": f"This is root!"}

# ENDPOINT: /gradio
app = gr.mount_gradio_app(app, gradio_app123, path="/gradio")

# ====================================================================================================

uvicorn.run(
    app, 
    host = "127.0.0.1" if not os.getenv("OS_ENV_DOCKER") else "0.0.0.0",
    port = 80,
)