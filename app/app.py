from fastapi.middleware.cors import CORSMiddleware as FAPI_CORSMiddleware
from fastapi.staticfiles import StaticFiles as FAPI_StaticFiles
from fastapi.responses import FileResponse as FAPI_FileResponse
from fastapi import FastAPI as FAPI_FastAPI
import gradio as gr
import uvicorn
import os

fapi = FAPI_FastAPI()
fapi.add_middleware(FAPI_CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ====================================================================================================

# ENDPOINT: /root
fapi.mount("/root_static", FAPI_StaticFiles(directory="app/root/root_static"))
@fapi.get("/")
def endpoint_root():
    return FAPI_FileResponse('app/root/index.html')

# ENDPOINT: /gradio
gradio_app123 = gr.Interface(lambda aaaaa: "Hello, " + aaaaa + "!", "textbox", "textbox")
fapi = gr.mount_gradio_app(fapi, gradio_app123, path="/gradio")

# ====================================================================================================

uvicorn.run(
    fapi, 
    host = "127.0.0.1" if not os.getenv("OS_ENV_DOCKER") else "0.0.0.0",
    port = 80,
)