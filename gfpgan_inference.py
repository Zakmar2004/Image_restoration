import os
import cv2
import torch
import numpy as np
from PIL import Image
from basicsr.utils import imwrite
from gfpgan import GFPGANer

def enhance_with_gfpgan(input_image_path, upscale=2):
    # Параметры модели
    version = '1.4'  # Версия модели
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.4'
    model_path = os.path.abspath("GFPGAN/gfpgan/weights/GFPGANv1.4.pth")  # Укажите путь к модели
    bg_upsampler = None  # Можно настроить, если нужно улучшение фона

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

    # Восстановление лиц
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,  # Убедитесь, что ваше изображение выровнено или не выровнено
        only_center_face=False,
        paste_back=True,
        weight=0.5
    )

    # Сохранение изображений
    temp_dir = '/tmp/restored_imgs'  # Путь для временных файлов
    os.makedirs(temp_dir, exist_ok=True)

    # Сохранение восстановленных изображений
    if restored_img is not None:
        save_restore_path = os.path.join(temp_dir, 'restored_image.png')
        imwrite(restored_img, save_restore_path)

    # Сохранение отдельных лиц
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        save_crop_path = os.path.join(temp_dir, f'cropped_face_{idx}.png')
        imwrite(cropped_face, save_crop_path)

        save_restore_path = os.path.join(temp_dir, f'restored_face_{idx}.png')
        imwrite(restored_face, save_restore_path)

    # Открытие и возврат восстановленного изображения
    enhanced_image = Image.open(save_restore_path)

    return enhanced_image









