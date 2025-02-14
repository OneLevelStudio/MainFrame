from fastapi.middleware.cors import CORSMiddleware as FAPI_CORSMiddleware
from fastapi.staticfiles import StaticFiles as FAPI_StaticFiles
from fastapi.responses import FileResponse as FAPI_FileResponse
from fastapi import FastAPI as FAPI_FastAPI
import gradio as gr
import uvicorn
import os

os.environ["OS_ENV_FASTAPI_WRAPPER"] = "TRUE"

fapi = FAPI_FastAPI()
fapi.add_middleware(FAPI_CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ====================================================================================================

# ENDPOINT: /
fapi.mount("/static_root", FAPI_StaticFiles(directory="app/root/static_root"))
@fapi.get("/")
def endpoint_root():
    return FAPI_FileResponse('app/root/index.html')

# ENDPOINT: /app1
from app1.app1 import demo as app1_demo
fapi = gr.mount_gradio_app(app=fapi, blocks=app1_demo, path="/app1")

# ENDPOINT: /app2
from app2.app2 import demo as app2_demo
fapi = gr.mount_gradio_app(app=fapi, blocks=app2_demo, path="/app2")

# ENDPOINT: /ocr
fapi.mount("/static_ocr", FAPI_StaticFiles(directory="app/ocr/static_ocr"))
from ocr.ocr import demo as ocr_demo
fapi = gr.mount_gradio_app(app=fapi, blocks=ocr_demo, path="/ocr")

# ====================================================================================================

uvicorn.run(
    fapi, 
    host = "127.0.0.1" if not os.getenv("OS_ENV_DOCKER_WRAPPER") else "0.0.0.0",
    port = 80,
)