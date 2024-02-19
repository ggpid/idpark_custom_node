from server import PromptServer
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import comfy
from PIL import Image
from tools import run_length_decode, run_length_encode
import time


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
        start = time.time()

        if sam_model.is_auto_mode:
            device = comfy.model_management.get_torch_device()
            sam_model.safe_to.to_device(sam_model, device=device)

        numpy_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

        print(f'loaded {time.time() - start}')

        masks_combined = mask_generator.generate(numpy_image)

        print(f'generated {time.time() - start}')

        masks = [mask["segmentation"] for mask in masks_combined]
        masks_encoded = [run_length_encode(mask) for mask in masks]

        print(f'encoded {time.time() - start}')

        message = {"node": "SAMGenerator", "output": {"masks": masks_encoded}}
        PromptServer.instance.send_sync("executed", message)

        print(f'synced {time.time() - start}')

        return {"ui": {"masks": masks_encoded}}
