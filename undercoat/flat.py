from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import copy
from PIL import Image
import torch
import cv2
from undercoat.convertor import rgba2df, mask2df, pil2cv, df2rgba
from tqdm import tqdm


def get_mask_generator(pred_iou_thresh, stability_score_thresh, min_mask_region_area, model_path, exe_mode):

    sam_checkpoint = f"{model_path}/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"

    if exe_mode == "extension":
        from modules.safe import unsafe_torch_load, load        
        torch.load = unsafe_torch_load
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        torch.load = load
    else:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            points_per_batch=32
        )
    
    return mask_generator

def get_masks(image, mask_generator):
    masks = mask_generator.generate(image)
    return masks

def mode_fast(series):
    return series.mode().iloc[0]

def show_anns(image, masks):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    mask_list = []
    for mask in sorted_masks:
        m = mask['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        img = np.dstack((img*255, m*255*0.35))
        img = img.astype(np.uint8)
        
        mask_list.append(img)
    
    base_mask = image 
    for mask in mask_list:
        base_mask = Image.alpha_composite(base_mask, Image.fromarray(mask))

    return base_mask

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)

def get_seg_base(input_image, masks, th):
    df = rgba2df(input_image)
    df["label"] = -1
    for idx, mask in tqdm(enumerate(masks)):
        if int(mask["area"] < th):
            continue
        mask_df = mask2df(mask["segmentation"])
        df = df.merge(mask_df, left_on=["x_l", "y_l"], right_on=["x_l_m", "y_l_m"], how="inner")
        df["label"] = np.where(df["m_flg"] == True, idx, df["label"])
        df.drop(columns=["x_l_m", "y_l_m", "m_flg"], inplace=True)

    df['r'] = df.groupby('label')['r'].transform(mode_fast)
    df['g'] = df.groupby('label')['g'].transform(mode_fast)
    df['b'] = df.groupby('label')['b'].transform(mode_fast)
    return df

def split_img_df(df, show=False):
    img_list = []
    for cls_no in tqdm(list(df["label"].unique())):
        img_df = df.copy()
        img_df.loc[df["label"] != cls_no, ["a"]] = 0 
        df_img = df2rgba(img_df).astype(np.uint8)
        img_list.append(df_img)
    return img_list

def segment(model_dir, gen_image):
    pred_iou_thresh = 0.9
    stability_score_thresh = 0.9 
    min_mask_region_area = 10000
    mask_generator = get_mask_generator(pred_iou_thresh, stability_score_thresh, min_mask_region_area, model_dir, "demo")
    masks = get_masks(pil2cv(gen_image), mask_generator)
   
    return masks

def get_line_img(rgba):
    white_pixels = (rgba[..., :3] >= [200, 200, 200]).all(axis=2)
    rgba[white_pixels, 3] = 0
    return rgba
    
    

def get_flat_img(gen_image, masks):
    gen_image.putalpha(255)
    image = pil2cv(gen_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    df = get_seg_base(image, masks, 1000)
    freq_colors = df.groupby('label')['r', 'g', 'b'].agg(lambda x: x.value_counts().index[0])

    for label, color in freq_colors.iterrows():
        df.loc[df['label'] == label, ['r', 'g', 'b']] = color.values

    layer_list = split_img_df(df)
    return df2rgba(df).astype(np.uint8), layer_list

