import os

import folder_paths
from server import PromptServer
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from fastsam import FastSAM
import numpy as np
import torch
import comfy
from PIL import Image
from tools import run_length_encode, resize_mask, resize_mask_centered


class SAMGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ()
    # RETURN_NAMES = ("masks",)

    FUNCTION = "doIt"

    OUTPUT_NODE = True

    CATEGORY = "CODEWAVE"

    def doIt(self, sam_model, image):
        numpy_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        sam_model.to('cuda')
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
        )
        masks_combined = mask_generator.generate(numpy_image)

        masks = [mask["segmentation"] for mask in masks_combined]
        masks_encoded = [run_length_encode(mask) for mask in masks]
        return {"ui": {"masks": masks_encoded}}

class FastSAMGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required":
                {"image": (sorted(files), {"image_upload": True})},
        }

    RETURN_TYPES = ()
    # RETURN_NAMES = ("masks",)

    FUNCTION = "doIt"

    OUTPUT_NODE = True

    CATEGORY = "CODEWAVE"

    def doIt(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        input = Image.open(image_path)
        input = input.convert("RGB")
        fastsam_model = FastSAM('./FastSAM-x.pt')
        everything_results = fastsam_model(
            input,
            device='cuda'
        )
        n_masks = everything_results[0].masks.data.cpu().numpy().squeeze()
        masks_encoded = []
        for mask in n_masks:
            mask_resized = resize_mask_centered(mask, input.width, input.height)
            mask_encoded = run_length_encode(mask_resized)
            masks_encoded.append(mask_encoded)

        return {"ui": {"masks": masks_encoded}}
