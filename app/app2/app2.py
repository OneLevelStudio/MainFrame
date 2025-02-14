import gradio as gr
import os

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

if not os.getenv("OS_ENV_FASTAPI_WRAPPER"):
    demo.launch(
        server_name="127.0.0.1" if not os.getenv("OS_ENV_DOCKER_WRAPPER") else "0.0.0.0",
        server_port=6969,
    )