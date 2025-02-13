import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gradio as gr
import os

# ====================================================================================================

from lib.VIOCR.VIOCR import Process_VIOCR as VIOCR
def main_func(input_text):
    ocr_text = VIOCR("example/test.jpg", bbox_padding=5, same_line_max_bbox_y_diff=10, new_line_char="\n", same_line_char=" | ", print_debug=False)
    return f"{ocr_text}"

# ====================================================================================================

with gr.Blocks() as demo:
    comp_input = gr.Textbox(label="Input Box")
    comp_output = gr.Textbox(label="Output Box")
    comp_button = gr.Button("Submit")
    comp_button.click(fn=main_func, inputs=comp_input, outputs=comp_output)

# ====================================================================================================

demo.launch(
    server_name="127.0.0.1" if not os.getenv("OS_ENV_DOCKER") else "0.0.0.0",
    server_port=6969,
)