import os
import json
import tempfile
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import uuid

from ..client_s3 import get_s3_instance
S3_INSTANCE = get_s3_instance()


class SaveImageS3:
    def __init__(self):
        self.s3_output_dir = os.getenv("S3_OUTPUT_DIR")
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "object_id": ("STRING", {"default": "${prompt_id}"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "CODEWAVE"

    def save_and_upload_image(self, image, object_id, index, format="PNG", quality=80):
        """
        이미지를 주어진 형식으로 저장하고 S3에 업로드하는 함수.
        """
        # 파일 확장자 및 저장 옵션 설정
        if format.upper() == "WEBP":
            filename = f"{index}.webp"
            save_args = {"format": format, "quality": quality}
        else:  # Default to PNG
            filename = f"{index}.png"
            save_args = {"format": "PNG", "compress_level": self.compress_level}

        # 임시 파일 생성 및 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format.lower()}") as temp_file:
            temp_file_path = temp_file.name
            image.save(temp_file_path, **save_args)

            # S3에 파일 업로드
            s3_path = f'{self.s3_output_dir}/{object_id}/{filename}'
            S3_INSTANCE.upload_file(temp_file_path, s3_path)

        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return s3_path


    def save_images(self, images, object_id):
        results = list()

        for index, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # PNG 형식으로 저장 및 업로드
            png_s3_path = self.save_and_upload_image(img, object_id, index, format="PNG")
            results.append(png_s3_path)

            # WEBP 형식으로 저장 및 업로드
            webp_s3_path = self.save_and_upload_image(img, object_id, index, format="WEBP", quality=80)
            # results.append(webp_s3_path)

        return { "ui": { "image_urls": results } }


