import numpy as np
import matplotlib.pyplot as plt

from pytoshop import layers
from pytoshop.user import nested_layers
import pytoshop

from PIL import Image

import random, string
import os

import psd_tools
from psd_tools.psd import PSD

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