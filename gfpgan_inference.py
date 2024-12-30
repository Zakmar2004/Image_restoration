import subprocess
from PIL import Image
import tempfile
import os

def enhance_with_gfpgan(input_image_path, upscale=2):
    with tempfile.TemporaryDirectory() as temp_dir:
        command = [
            "python", "GFPGAN/inference_gfpgan.py",
            "-i", input_image_path,
            "-o", temp_dir,
            "-v", '1.4',
            "-s", str(upscale),
            "--bg_upsampler", "realesrgan"
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"GFPGAN output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during GFPGAN execution: {e.stderr}")
            raise RuntimeError(f"GFPGAN execution failed: {e.stderr}") from e

        restored_img_dir = os.path.join(temp_dir, 'restored_imgs')
        if not os.path.exists(restored_img_dir):
            raise FileNotFoundError(f"Restored image directory not found at {restored_img_dir}")

        # Поиск первого изображения в выходной директории
        result_image_files = [f for f in os.listdir(restored_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not result_image_files:
            raise FileNotFoundError(f"No images found in restored image directory: {restored_img_dir}")

        result_image_path = os.path.join(restored_img_dir, result_image_files[0])

        # Открытие изображения
        try:
            enhanced_image = Image.open(result_image_path)
        except Exception as e:
            raise IOError(f"Failed to open the enhanced image: {e}")

        return enhanced_image








