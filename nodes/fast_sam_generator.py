import folder_paths
from fastsam import FastSAM
from PIL import Image
import numpy as np
from ..tools import run_length_encode, resize_mask, resize_mask_centered
import os


class FastSAMGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ()
    # RETURN_NAMES = ("masks",)

    FUNCTION = "doIt"

    OUTPUT_NODE = True

    CATEGORY = "CODEWAVE"

    def doIt(self, images):
        image = images[0]
        i = 255. * image.cpu().numpy()
        input = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
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
