import gradio as gr
import sys
sys.path.append("./undercoat/")

from undercoat.sd_model import get_cn_pipeleine, get_cn_detector, generate
from undercoat.flat import segment, get_line_img, get_flat_img
from undercoat.convertor import pil2cv, cv2pil
from undercoat.utils import save_psd, load_seg_model

import cv2
from pytoshop.enums import BlendMode
import os

import numpy as np
from PIL import Image

path = os.getcwd()
output_dir = f"{path}/output"
input_dir = f"{path}/input"
model_dir = f"{path}/segment_model"

load_seg_model(model_dir)

class webui:
    def __init__(self):
        self.demo = gr.Blocks()

    def undercoat(self, input_image, pos_prompt, neg_prompt, bg_type):
        image = pil2cv(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        if bg_type == "alpha":
            line_img = pil2cv(input_image)
            line_img = cv2.cvtColor(line_img, cv2.COLOR_BGRA2RGBA)
            index = np.where(image[:, :, 3] == 0)
            image[index] = [255, 255, 255, 255]
            input_image = cv2pil(image)
        else:
            line_img = get_line_img(image)

        pipe = get_cn_pipeleine()
        detectors = get_cn_detector(input_image)
            
        gen_image = generate(pipe, detectors, pos_prompt, neg_prompt)

        masks = segment(model_dir, pil2cv(gen_image))
        output, layer_list = get_flat_img(gen_image, masks)
        layer_list.append(line_img)

        layers = []

        for layer in layer_list:
            layers.append(cv2.resize(layer, (line_img.shape[1], line_img.shape[0])))

        output = cv2.resize(output, (line_img.shape[1], line_img.shape[0]))

        filename = save_psd(
            line_img,
            [layers],
            ["base"],
            [BlendMode.normal],
            output_dir,
        )

        output = cv2pil(output)
        line_img = cv2pil(line_img)

        output = Image.alpha_composite(output, line_img)
        output = pil2cv(output)

        return output, layers, filename



    def launch(self, share):
        with self.demo:
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil")
                    bg_type = gr.Dropdown(["alpha", "white"], value = "alpha", label="bg_type", show_label=True)

                    pos_prompt = gr.Textbox(max_lines=1000, label="positive prompt")                    
                    neg_prompt = gr.Textbox(max_lines=1000, label="negative prompt")

                    submit = gr.Button(value="Create PSD")
                with gr.Row():
                    with gr.Column():
                        with gr.Tab("output"):
                            output_0 = gr.Image()
                        with gr.Tab("layers"):
                            output_1 = gr.Gallery()

                        output_file = gr.File()
                    
            submit.click(
                self.undercoat, 
                inputs=[input_image, pos_prompt, neg_prompt, bg_type], 
                outputs=[output_0, output_1, output_file]
            )

        self.demo.queue()
        self.demo.launch(share=share)


if __name__ == "__main__":
    ui = webui()
    if len(sys.argv) > 1:
        if sys.argv[1] == "share":
            ui.launch(share=True)
        else:
            ui.launch(share=False)
    else:
        ui.launch(share=False)
