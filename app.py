import gradio as gr
import os

def main_func(s):
    return "Hello " + s + "!"

with gr.Blocks() as demo:
    comp_input = gr.Textbox(label="Input")
    comp_output = gr.Textbox(label="Output")
    comp_button = gr.Button("Submit")
    comp_button.click(fn=main_func, inputs=comp_input, outputs=comp_output)

demo.launch(
    server_name="127.0.0.1" if not os.getenv("OS_ENV_DOCKER") else "0.0.0.0",
    server_port=6969,
)