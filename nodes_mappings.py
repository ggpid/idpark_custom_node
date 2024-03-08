from .nodes.load_image_s3 import LoadImageS3
from .nodes.save_image_s3 import SaveImageS3
from .nodes.sam_generator import SAMGenerator
from .nodes.fast_sam_generator import FastSAMGenerator
from .nodes.cut_by_mask_fixed import CutByMaskFixed


NODE_CLASS_MAPPINGS = {
    "LoadImageS3": LoadImageS3,
    "SaveImageS3": SaveImageS3,
    "SAMGenerator": SAMGenerator,
    "FastSAMGenerator": FastSAMGenerator,
    "CutByMaskFixed": CutByMaskFixed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageS3": "Load Image from S3",
    "SaveImageS3": "Save Image to S3",
    "SAMGenerator": "Generate SAM",
    "FastSAMGenerator": "Generate FastSAM",
    "CutByMaskFixed": "Cut by Mask fixed",
}