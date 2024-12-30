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
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr.decode()}")
            return None

        restored_img_dir = os.path.join(temp_dir, 'restored_imgs')
        result_image_path = os.path.join(restored_img_dir, 'input_image.jpg')
        enhanced_image = Image.open(result_image_path)

        return enhanced_image







