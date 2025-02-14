import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ====================================================================================================

from lib.VIOCR.VIOCR import Process_VIOCR as VIOCR
def fn_image_upload(img):
    ocr_text = VIOCR(img, bbox_padding=5, same_line_max_bbox_y_diff=10, new_line_char="\n", same_line_char=" | ", print_debug=False)
    return ocr_text
def fn_image_clear(img):
    return ""

with gr.Blocks(
    title="OCR", 
    theme=gr.themes.Base(
        primary_hue="rose",
        secondary_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont('Inter'), 'monospace'],
        font_mono=[gr.themes.GoogleFont('JetBrains Mono'), 'monospace'],
    ),
    head="""
        <link rel="icon" href="/static_ocr/favicon.png">
    """,
    css="""
        footer {
            gap: 16px;
        }
        footer .built-with,
        footer .divider {
            display: none !important;
        }
    """,
    fill_height=True,
    fill_width=True,
) as demo:
    with gr.Row():
        with gr.Column():
            comp_input = gr.Image(type="filepath", height="501px", show_label=False, show_fullscreen_button=True, placeholder="Drop Image Here\nor\nClick to Upload")
        with gr.Column():
            comp_output = gr.Textbox(lines=22, label="OCR Text", show_copy_button=True, interactive=True)
    # Event Listeners
    event_image_upload = comp_input.upload(fn=fn_image_upload, inputs=comp_input, outputs=comp_output, api_name="ocr")
    comp_input.clear(fn=fn_image_clear, inputs=comp_input, outputs=comp_output, cancels=[event_image_upload], api_name=False)

# ====================================================================================================

if not os.getenv("OS_ENV_FASTAPI_WRAPPER"):
    demo.launch(
        server_name="127.0.0.1" if not os.getenv("OS_ENV_DOCKER_WRAPPER") else "0.0.0.0",
        server_port=6969,
    )