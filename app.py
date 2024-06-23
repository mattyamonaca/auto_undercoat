import gradio as gr
import sys
from starline import process

from utils import load_cn_model, load_cn_config, randomname, load_lora_model
from convertor import pil2cv, cv2pil, df2bgra

from sd_model import get_cn_pipeline, generate, get_cn_detector, get_ip_pipeline
import cv2
import os
import numpy as np
from PIL import Image
import zipfile
from outerline import interpolation

path = os.getcwd()
output_dir = f"{path}/output"
input_dir = f"{path}/input"
cn_lineart_dir = f"{path}/controlnet/lineart"
lora_dir = f"{path}/lora"

load_cn_model(cn_lineart_dir)
load_cn_config(cn_lineart_dir)
load_lora_model(lora_dir)

pipe_cn = get_cn_pipeline()
pipe_cn.to("cuda")

"""IPAdapterを使用する場合はVRAM24GB以上を推奨
pipe_ip = get_ip_pipeline() 
pipe_ip.to("cuda")
"""




def resize_image(img, max_size=1024):
    # 画像を開く
    width, height = img.size
    print(f"元の画像サイズ: 幅 {width} x 高さ {height}")
    
    # 縦または横がmax_sizeを超えているかチェック
    if width > max_size or height > max_size:
        # 縦横比を保ちながらリサイズ
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        
        # リサイズ実行
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        print(f"リサイズ後の画像サイズ: 幅 {new_width} x 高さ {new_height}")
        return resized_img
    else:
        return img



def composite_images(image_list):
    # リストが空か、画像が1つしかない場合は、その画像をそのまま返す
    if not image_list or len(image_list) == 1:
        return image_list[0] if image_list else None

    # リストの最初の画像をベースとする
    base_image = image_list[0]

    # 2番目の画像から順に重ねていく
    for image in image_list[1:]:
        base_image = Image.alpha_composite(base_image, image)

    return base_image

def generate(detectors, prompt, negative_prompt, reference_img=None):
    default_pos = "bestquality, 4K, flatcolor, (sdxl-flat:1)"
    default_neg = "shadow, (worst quality, low quality:1.2), (lowres:1.2), (bad anatomy:1.2), (greyscale, monochrome:1.4)"
    prompt = default_pos + prompt 
    negative_prompt = default_neg + negative_prompt


    gen_image = pipe_cn(
                    prompt=prompt,
                    negative_prompt = negative_prompt,
                    image=detectors,
                    num_inference_steps=50,
                    controlnet_conditioning_scale=[1.0, 0.2]
                ).images[0]

    """IPAdapterを使用する場合はVRAM24GB以上を推奨
    if reference_img is not None:
        gen_image = pipe_ip(
                    prompt=prompt,
                    negative_prompt = negative_prompt,
                    image=detectors,
                    num_inference_steps=50,
                    controlnet_conditioning_scale=[1.0, 0.2],
                    ip_adapter_image=reference_img,
                ).images[0]
    else:

        gen_image = pipe_cn(
                    prompt=prompt,
                    negative_prompt = negative_prompt,
                    image=detectors,
                    num_inference_steps=50,
                    controlnet_conditioning_scale=[1.0, 0.2]
                ).images[0]
    """
        
    return gen_image


def zip_png_files(folder_path, name):
    # Zipファイルの名前を設定（フォルダ名と同じにします）
    zip_path = os.path.join(folder_path, f'{name}.zip')
    
    # zipfileオブジェクトを作成し、書き込みモードで開く
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # フォルダ内のすべてのファイルをループ処理
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                # PNGファイルのみを対象にする
                if filename.endswith('.png'):
                    # ファイルのフルパスを取得
                    file_path = os.path.join(foldername, filename)
                    # zipファイルに追加
                    zipf.write(file_path, arcname=os.path.relpath(file_path, folder_path))


class webui:

    def __init__(self):
        self.demo = gr.Blocks()

    def undercoat(self, input_image, pos_prompt, neg_prompt, alpha_th, thickness, reference_img=None):
        input_image = resize_image(input_image)
        

        org_line_image = input_image
        image = pil2cv(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        index = np.where(image[:, :, 3] == 0)
        image[index] = [255, 255, 255, 255]
        input_image = cv2pil(image)

        detectors = get_cn_detector(input_image.resize((1024, 1024), Image.ANTIALIAS))

        gen_image = generate(detectors, pos_prompt, neg_prompt)
        color_img, unfinished, images, unfinished_color = process(gen_image.resize((image.shape[1], image.shape[0]), Image.ANTIALIAS) , org_line_image, alpha_th, thickness)
        
        name = randomname(10)
        os.makedirs(f"{output_dir}/{name}")
        interpolated_list = []


        print("start interpolation")
        print(unfinished_color)
        for idx, img in enumerate(images):
            print(img[1])

            if img[1] == unfinished_color:
                interpolated_list.append(img[0])
                img[0].save(f"{output_dir}/{name}/area_{idx}.png")
                continue
            interpolated_img = interpolation(img[0], img[1])
            interpolated_list.append(interpolated_img)
            interpolated_img.save(f"{output_dir}/{name}/area_{idx}.png")

        
        org_line_image.save(f"{output_dir}/{name}/line_image.png")
        unfinished.save(f"{output_dir}/{name}/unfinished_image.png")

        flat_image = composite_images(interpolated_list) 
        flat_image.save(f"{output_dir}/{name}/color_image.png")
        
        output_img = Image.alpha_composite(flat_image, org_line_image)
        output_img.save(f"{output_dir}/{name}/output_image.png")

        outputs = [output_img, org_line_image, flat_image, unfinished]
        zip_png_files(f"{output_dir}/{name}", name)
        filename = f"{output_dir}/{name}/{name}.zip"

        return outputs, filename



    def launch(self, share):
        with self.demo:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(type="pil", image_mode="RGBA", label="lineart")
                        #reference_image = gr.Image(type="pil", image_mode="RGB", label="reference_image") #IPAdapter使用時はコメントアウトを外す

                    pos_prompt = gr.Textbox(max_lines=1000, label="positive prompt")                    
                    neg_prompt = gr.Textbox(max_lines=1000, label="negative prompt")

                    alpha_th = gr.Slider(maximum = 255, value=100, label = "alpha threshold")
                    thickness = gr.Number(value=5, label="Thickness of correction area (Odd numbers need to be entered)")
                    #gr.Slider(maximum = 21, value=3, step=2, label = "Thickness of correction area")

                    submit = gr.Button(value="Start")

                with gr.Row():
                    with gr.Column():
                        with gr.Tab("output"):
                            output_0 = gr.Gallery(format="png")
                        output_file = gr.File()
                
                
            submit.click(
                self.undercoat, 
                inputs=[input_image, pos_prompt, neg_prompt, alpha_th, thickness], #[input_image, pos_prompt, neg_prompt, alpha_th, thickness, reference_image], 
                outputs=[output_0, output_file]
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
