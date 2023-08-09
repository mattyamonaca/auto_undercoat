from pytoshop import layers
import pytoshop

import random
import string
import os

import requests
from tqdm import tqdm


def add_psd(psd, img, name, mode):

  layer_1 = layers.ChannelImageData(image=img[:, :, 3], compression=1)
  layer0 = layers.ChannelImageData(image=img[:, :, 0], compression=1)
  layer1 = layers.ChannelImageData(image=img[:, :, 1], compression=1)
  layer2 = layers.ChannelImageData(image=img[:, :, 2], compression=1)

  new_layer = layers.LayerRecord(channels={-1: layer_1, 0: layer0, 1: layer1, 2: layer2},
                                  top=0, bottom=img.shape[0], left=0, right=img.shape[1],
                                  blend_mode=mode,
                                  name=name,
                                  opacity=255,
                                  )
  psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
  return psd

def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)

def save_psd(input_image, layers, names, modes, output_dir):
    psd = pytoshop.core.PsdFile(num_channels=3, height=input_image.shape[0], width=input_image.shape[1])
    for idx, output in enumerate(layers[0]):
        psd = add_psd(psd, layers[0][idx], names[0] + str(idx), modes[0])
    name = randomname(10)

    with open(f"{output_dir}/output_{name}.psd", 'wb') as fd2:
        psd.write(fd2)

    return f"{output_dir}/output_{name}.psd"

def load_seg_model(model_dir):
  folder = model_dir
  file_name = 'sam_vit_h_4b8939.pth'
  url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
