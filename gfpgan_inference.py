import os
import cv2
import torch
import numpy as np
from PIL import Image
from basicsr.utils import imwrite
from gfpgan import GFPGANer

def enhance_with_gfpgan(input_image_path, upscale=2):
    version = '1.4'
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.4'
    model_path = os.path.abspath("GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth")
    bg_upsampler = None

    # Загрузка модели
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler
    )

    # Чтение изображения
    input_img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    # Восстановление лиц и всего изображения
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,  # Если изображение не выровнено
        only_center_face=False,  # Восстановить все лица
        paste_back=True,  # Вставить восстановленные лица обратно на изображение
        weight=0.5  # Влияние восстановления на исходное изображение
    )

    # Если изображение было успешно восстановлено, возвращаем его
    if restored_img is not None:
        enhanced_image = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        return enhanced_image

    return None  # Возвращаем None, если восстановленное изображение отсутствует










