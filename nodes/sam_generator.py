import numpy as np
from segment_anything import SamAutomaticMaskGenerator
from ..tools import run_length_encode


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
