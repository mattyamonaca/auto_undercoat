from PIL import Image, ImageFilter
from collections import defaultdict
from skimage import color as sk_color
from PIL import Image
from tqdm import tqdm
from skimage.color import deltaE_ciede2000, rgb2lab
import cv2
from collections import Counter
import numpy as np

from PIL import Image, ImageOps
from scipy.ndimage import label



def modify_transparency(img, target_rgb):
    # 画像を読み込む
    copy_img = img.copy() 
    data = copy_img.getdata()

    # 新しいピクセルデータを作成
    new_data = []
    for item in data:
        # 指定されたRGB値のピクセルの場合、透明度を255に設定
        if item[:3] == target_rgb:
            new_data.append((item[0], item[1], item[2], 255))
        else:
            # それ以外の場合、透明度を0に設定
            new_data.append((item[0], item[1], item[2], 0))

    # 新しいデータを画像に設定し直す
    copy_img.putdata(new_data)
    return copy_img


def replace_color(image, color_1, color_2, alpha_np):
    # 画像データを配列に変換
    data = np.array(image)

    # RGBAモードの画像であるため、形状変更時に4チャネルを考慮
    original_shape = data.shape
    data = data.reshape(-1, 4)  # RGBAのため、4チャネルでフラット化

    # color_1のマッチングを検索する際にはRGB値のみを比較
    matches = np.all(data[:, :3] == color_1, axis=1)

    # 変更を追跡するためのフラグ
    nochange_count = 0
    idx = 0

    while np.any(matches):
        idx += 1
        new_matches = np.zeros_like(matches)
        match_num = np.sum(matches)
        for i in tqdm(range(len(data))):
            if matches[i]:
                x, y = divmod(i, original_shape[1])
                neighbors = [
                    (x-1, y), (x+1, y), (x, y-1), (x, y+1)  # 上下左右
                ]
                replacement_found = False
                for nx, ny in neighbors:
                    if 0 <= nx < original_shape[0] and 0 <= ny < original_shape[1]:
                        ni = nx * original_shape[1] + ny
                        # RGBのみ比較し、アルファは無視
                        if not np.all(data[ni, :3] == color_1, axis=0) and not np.all(data[ni, :3] == color_2, axis=0):
                            data[i, :3] = data[ni, :3]  # RGB値のみ更新
                            replacement_found = True
                            continue
                if not replacement_found:
                    new_matches[i] = True
        matches = new_matches
        if match_num == np.sum(matches):
             nochange_count += 1
        if nochange_count > 5:
            break

    # 最終的な画像をPIL形式で返す
    data = data.reshape(original_shape)
    data[:, :, 3] = 255 - alpha_np
    return Image.fromarray(data, 'RGBA')

def recolor_lineart_and_composite(lineart_image, base_image, new_color, alpha_th):
    """
    Recolor an RGBA lineart image to a single new color while preserving alpha, and composite it over a base image.
    Args:
    lineart_image (PIL.Image): The lineart image with RGBA channels.
    base_image (PIL.Image): The base image to composite onto.
    new_color (tuple): The new RGB color for the lineart (e.g., (255, 0, 0) for red).

    Returns:
    PIL.Image: The composited image with the recolored lineart on top.
    """
    # Ensure images are in RGBA mode
    if lineart_image.mode != 'RGBA':
        lineart_image = lineart_image.convert('RGBA')
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')

    # Extract the alpha channel from the lineart image
    r, g, b, alpha = lineart_image.split()

    alpha_np = np.array(alpha)
    alpha_np[alpha_np < alpha_th] = 0
    alpha_np[alpha_np >= alpha_th] = 255
    
    new_alpha = Image.fromarray(alpha_np)

    # Create a new image using the new color and the alpha channel from the original lineart
    new_lineart_image = Image.merge('RGBA', (
        Image.new('L', lineart_image.size, int(new_color[0])),
        Image.new('L', lineart_image.size, int(new_color[1])),
        Image.new('L', lineart_image.size, int(new_color[2])),
        new_alpha
    ))

    # Composite the new lineart image over the base image
    composite_image = Image.alpha_composite(base_image, new_lineart_image)

    return composite_image, alpha_np


def thicken_and_recolor_lines(base_image, lineart, thickness=3, new_color=(0, 0, 0)):
    """
    Thicken the lines of a lineart image, recolor them, and composite onto another image,
    while preserving the transparency of the original lineart.

    Args:
    base_image (PIL.Image): The base image to composite onto.
    lineart (PIL.Image): The lineart image with transparent background.
    thickness (int): The desired thickness of the lines.
    new_color (tuple): The new color to apply to the lines (R, G, B).

    Returns:
    PIL.Image: The image with the recolored and thickened lineart composited on top.
    """
    # Ensure both images are in RGBA format
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')
    if lineart.mode != 'RGB':
        lineart = lineart.convert('RGBA')
    
    # Convert the lineart image to OpenCV format
    lineart_cv = np.array(lineart)

    # 各チャンネルを分離
    b, g, r, a = cv2.split(lineart_cv)
    
    # アルファ値の処理
    new_a = np.where(a == 0, 255, 255).astype(np.uint8)
    new_c = np.where(a == 0, 255, 0).astype(np.uint8)
        
    # 画像を再結合
    lineart_cv = cv2.merge((new_c, new_c, new_c, new_a))
    mask = cv2.inRange(lineart_cv, (0, 0, 0, 0), (0, 0, 0, 255))
    lineart_cv[mask == 0] = [255, 255, 255, 255]

    white_pixels = np.sum(lineart_cv == 255)
    black_pixels = np.sum(lineart_cv == 0)
    

    lineart_gray = cv2.cvtColor(lineart_cv, cv2.COLOR_RGBA2GRAY)
    
    #_, lineart_gray = cv2.threshold(lineart_gray, 1, 255, cv2.THRESH_BINARY)
    #lineart_gray = cv2.cvtColor(lineart_cv, cv2.COLOR_RGBA2GRAY)
    

    if white_pixels > black_pixels:
        lineart_gray = cv2.bitwise_not(lineart_gray)
    # Thicken the lines using OpenCV
    kernel = np.ones((thickness, thickness), np.uint8)
    lineart_thickened = cv2.dilate(lineart_gray, kernel, iterations=1)
    
    #lineart_thickened = cv2.bitwise_not(lineart_thickened)
    

    # Create a new RGBA image for the recolored lineart
    lineart_recolored = np.zeros_like(lineart_cv)
    lineart_recolored[:, :, :3] = new_color  # Set new RGB color
    lineart_recolored[:, :, 3] = np.where(lineart_thickened  < 250, 0, 255)  # Blend alpha with thickened lines
   

    # Convert back to PIL Image
    lineart_recolored_pil = Image.fromarray(lineart_recolored, 'RGBA')
    
    # Composite the thickened and recolored lineart onto the base image
    combined_image = Image.alpha_composite(base_image, lineart_recolored_pil)
    return combined_image


def generate_distant_colors(consolidated_colors, distance_threshold):
    """
    Generate new RGB colors that are at least 'distance_threshold' CIEDE2000 units away from given colors.

    Args:
    consolidated_colors (list of tuples): List of ((R, G, B), count) tuples.
    distance_threshold (float): The minimum CIEDE2000 distance from the given colors.

    Returns:
    list of tuples: List of new RGB colors that meet the distance requirement.
    """
    #new_colors = []
    # Convert the consolidated colors to LAB
    consolidated_lab = [rgb2lab(np.array([color], dtype=np.float32) / 255.0).reshape(3) for color, _ in consolidated_colors]

    # Try to find a distant color
    max_attempts = 10000
    for _ in range(max_attempts):
        # Generate a random color in RGB and convert to LAB
        random_rgb = np.random.randint(0, 256, size=3)
        random_lab = rgb2lab(np.array([random_rgb], dtype=np.float32) / 255.0).reshape(3)
        for base_color_lab in consolidated_lab:
            # Calculate the CIEDE2000 distance
            distance = deltaE_ciede2000(base_color_lab, random_lab)
            if distance <= distance_threshold:
                break
        new_color = tuple(random_rgb)
        break
    return new_color
                


def consolidate_colors(major_colors, threshold):
    """
    Consolidate similar colors in the major_colors list based on the CIEDE2000 metric.

    Args:
    major_colors (list of tuples): List of ((R, G, B), count) tuples.
    threshold (float): Threshold for CIEDE2000 color difference.

    Returns:
    list of tuples: Consolidated list of ((R, G, B), count) tuples.
    """
    # Convert RGB to LAB
    colors_lab = [rgb2lab(np.array([[color]], dtype=np.float32)/255.0).reshape(3) for color, _ in major_colors]
    n = len(colors_lab)

    # Find similar colors and consolidate
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            delta_e = deltaE_ciede2000(colors_lab[i], colors_lab[j])
            if delta_e < threshold:
                # Compare counts and consolidate to the color with the higher count
                if major_colors[i][1] >= major_colors[j][1]:
                    major_colors[i] = (major_colors[i][0], major_colors[i][1] + major_colors[j][1])
                    major_colors.pop(j)
                    colors_lab.pop(j)
                else:
                    major_colors[j] = (major_colors[j][0], major_colors[j][1] + major_colors[i][1])
                    major_colors.pop(i)
                    colors_lab.pop(i)
                n -= 1
                continue
            j += 1
        i += 1

    return major_colors




def get_major_colors(image, threshold_percentage=0.01):
    """
    Analyze an image to find the major RGB values based on a threshold percentage.

    Args:
    image (PIL.Image): The image to analyze.
    threshold_percentage (float): The percentage threshold to consider a color as major.

    Returns:
    list of tuples: A list of (color, count) tuples for colors that are more frequent than the threshold.
    """
    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Count each color
    color_count = defaultdict(int)
    for pixel in image.getdata():
        color_count[pixel] += 1

    # Total number of pixels
    total_pixels = image.width * image.height

    # Filter colors to find those above the threshold
    major_colors = [(color, count) for color, count in color_count.items()
                    if (count / total_pixels) >= threshold_percentage]

    return major_colors

def get_binary_image(image, target_rgb):
    copy_image = image.copy()
    pixels = copy_image.load()

    # 画像サイズを取得
    width, height = image.size
    rgb_list = list(target_rgb)
    rgb_list.append(255)
    target_rgb = tuple(rgb_list)
    print(target_rgb)

    # ピクセルごとに処理
    for y in range(height):
        for x in range(width):
            # 現在のピクセルのRGB値を取得
            current_rgb = pixels[x, y]
            # ターゲットのRGB値と一致する場合
            if current_rgb == target_rgb:
                # 黒に設定
                pixels[x, y] = (0, 0, 0)
            else:
                # 白に設定
                pixels[x, y] = (255, 255, 255)

    # 変更を保存
    return copy_image

def binarize_image(image, threshold=128):
    gray_image = ImageOps.grayscale(image)
    binary_image = gray_image.point(lambda x: 255 if x > threshold else 0, '1')
    binary_image.save("tmp_binary.png")
    return binary_image


def find_contours(binary_image):
    binary_array = np.array(binary_image, dtype=np.uint8)
    labeled_array, num_features = label(binary_array)
    return labeled_array, num_features

def get_most_frequent_color(image, labeled_array, label_id):
    mask = labeled_array == label_id
    pixels = [image.getpixel((x, y)) for y, x in np.argwhere(mask)]
    most_common_color = Counter(pixels).most_common(1)[0][0]
    return most_common_color

def fill_contours_with_color(image, labeled_array, num_features):
    for label_id in range(1, num_features + 1):
        most_frequent_color = get_most_frequent_color(image, labeled_array, label_id)
        for y, x in np.argwhere(labeled_array == label_id):
            image.putpixel((x, y), most_frequent_color)
    return image

def rgb_to_lab(color):
    if len(color) == 4:  # If RGBA, convert to RGB
        color = color[:3]
    rgb = np.array([[color]], dtype=np.float32) / 255.0
    lab = sk_color.rgb2lab(rgb)[0][0]
    return lab

def are_colors_similar(color1, color2, threshold=10):
    lab1 = rgb_to_lab(color1)
    lab2 = rgb_to_lab(color2)
    delta_e = sk_color.deltaE_cie76(lab1, lab2)  # CIEDE2000に変更
    return delta_e < threshold

def merge_similar_labels(labeled_array, colors, similarity_threshold=10):
    label_map = {label_id: label_id for label_id in colors}
    for label1, color1 in colors.items():
        for label2, color2 in colors.items():
            if label1 < label2 and are_colors_similar(color1, color2, similarity_threshold):
                for key in label_map:
                    if label_map[key] == label2:
                        label_map[key] = label1
    
    new_labeled_array = np.zeros_like(labeled_array)
    new_label_id_map = {}
    new_label_id = 1
    for old_label_id in np.unique(labeled_array):
        if old_label_id in label_map:
            mapped_label_id = label_map[old_label_id]
            if mapped_label_id not in new_label_id_map:
                new_label_id_map[mapped_label_id] = new_label_id
                new_label_id += 1
            new_labeled_array[labeled_array == old_label_id] = new_label_id_map[mapped_label_id]
    
    return new_labeled_array, len(new_label_id_map)

def extract_and_isolate_colors(image):
    original_image = image
    pixels = original_image.load()
    width, height = original_image.size

    # RGB値のセットを作成
    unique_colors = set()
    for x in range(width):
        for y in range(height):
            unique_colors.add(pixels[x, y][:3])

    # 各RGB値ごとに新しい画像を作成
    isolated_images = []
    for color in unique_colors:
        new_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        new_pixels = new_image.load()

        for x in range(width):
            for y in range(height):
                if pixels[x, y][:3] == color:
                    new_pixels[x, y] = pixels[x, y]

        isolated_images.append((new_image, color))

    return isolated_images


def find_closest_label(color, colors, exclude_labels):
    lab1 = rgb_to_lab(color)
    closest_label = None
    min_distance = float('inf')
    for label, color2 in colors.items():
        if label in exclude_labels:
            continue
        lab2 = rgb_to_lab(color2)
        delta_e = sk_color.deltaE_ciede2000(lab1, lab2)
        if delta_e < min_distance:
            min_distance = delta_e
            closest_label = label
    return closest_label


def merge_small_labels(image, labeled_array, colors, min_pixels=20):
    unique_labels, counts = np.unique(labeled_array, return_counts=True)
    small_labels = unique_labels[counts <= min_pixels]
    for label in tqdm(small_labels):
        if label == 0:
            continue
        color = colors[label]
        closest_label = find_closest_label(color, {k: v for k, v in colors.items() if k != label}, small_labels)
        if closest_label is not None:
            labeled_array[labeled_array == label] = closest_label
            colors[closest_label] = get_most_frequent_color(image, labeled_array, closest_label)
    return labeled_array

def process(image, lineart, alpha_th, thickness):
    org = image
    major_colors = get_major_colors(image, threshold_percentage=0.05) #主要な色を取得
    major_colors = consolidate_colors(major_colors, 10) #主要な色のうち、近しい色を統合
    new_color_1 = generate_distant_colors(major_colors, 100) #修正領域を表す色を生成
    image = thicken_and_recolor_lines(org, lineart, thickness=thickness, new_color=new_color_1) #線を太くして元画像に貼り付け
    tmp = get_binary_image(image, new_color_1) #太くした線のみを抽出
    binary_image = binarize_image(tmp) #太くした線のみを抽出
    labeled_array, num_features = find_contours(binary_image) #閉域を検出
    print(f"num features: {num_features}")

    #検出した閉域の最頻色を取得
    colors = {label_id: get_most_frequent_color(image, labeled_array, label_id)
        for label_id in range(1, num_features + 1)}

    #unique_labels, counts = np.unique(labeled_array, return_counts=True)
    labeled_array = merge_small_labels(image, labeled_array, colors, 1000) #ピクセル数が少ない領域を統合
    
    #unique_labels, counts = np.unique(labeled_array, return_counts=True)

    merged_labeled_array, merged_num_features = merge_similar_labels(labeled_array, colors, 10) #色が近い領域を統合

    flat_image = fill_contours_with_color(image.copy(), merged_labeled_array, merged_num_features) #閉域を最頻色で塗りつぶし
    major_colors.append((new_color_1, 0))

    #以下Starlineと同様の処理
    new_color_2 = generate_distant_colors(major_colors, 100)
    image, alpha_np = recolor_lineart_and_composite(lineart, flat_image, new_color_2, alpha_th)
    
    image = replace_color(image, new_color_1, new_color_2, alpha_np)
    images = extract_and_isolate_colors(image)
    unfinished = modify_transparency(image, new_color_1)

    return image, unfinished, images, new_color_1
"""
lineart = Image.open("./output/P4eqJpIBVS/line_image.png")
image = Image.open("./output/P4eqJpIBVS/color_image.png")

image, unfinished, images = process(image, lineart, 100, 5)

image.save("tmp_result_all.png")
image.save("tmp_unfinished.png")

ct = 0
for img in images:
    ct = ct + 1
    img.save(f"tmp_result_{ct}.png")
"""