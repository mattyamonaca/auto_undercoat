from PIL import Image, ImageOps
import numpy as np
from skimage import measure, morphology
from scipy.stats import mode

from tqdm import tqdm


def interpolation(image, color):
    # numpy配列に変換
    data = np.array(image)

    # α値が1以上のピクセルを検出（2値化）
    alpha = data[:, :, 3]
    binary_image = alpha > 0

    if np.sum(binary_image) == 0:
        return image

    # ラベリングして領域を検出
    labels = measure.label(binary_image)
    regions = measure.regionprops(labels, intensity_image=data[:, :, :3])

    # 各領域の最頻値のRGB値を取得
    """
    region_labels = []
    for region in tqdm(regions):
        # リージョンのRGB値を抽出
        region_pixels = data[:, :, :3] #[region.coords[:, 0], region.coords[:, 1]]
        # 最頻値を計算
        most_common_color = mode(region_pixels, axis=0).mode[0]
        region_colors[region.label] = most_common_color
    """

    # 外周を2ピクセル広げる処理
    dilated_labels = morphology.dilation(labels, morphology.disk(2))

    # 領域ごとに色を設定
    colored_dilated_image = data.copy()
    for region in tqdm(regions):
        label = region.label
        mask = dilated_labels == label
        colored_dilated_image[mask] = np.concatenate([color, [255]])  # RGB + α

    # 新しい画像を保存
    new_image = Image.fromarray(colored_dilated_image)
#    new_image_path = image_path.replace(".png", "_colored_processed.png")
#    new_image.save(new_image_path)

    return new_image
