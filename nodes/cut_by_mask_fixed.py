import torch
from ..tools import tensor2mask, tensor2rgba, tensor2rgb
from torchvision.ops import masks_to_boxes

class CutByMaskFixed:
    """
    Cuts the image to the bounding box of the mask. If force_resize_width or force_resize_height are provided, the image will be resized to those dimensions. The `mask_mapping_optional` input can be provided from a 'Separate Mask Components' node to cut multiple pieces out of a single image in a batch.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cut"

    CATEGORY = "CODEWAVE"

    def cut(self, image, mask, mask_mapping_optional=None):
        if len(image.shape) < 4:
            C = 1
        else:
            C = image.shape[3]

        # We operate on RGBA to keep the code clean and then convert back after
        image = tensor2rgba(image)
        mask = tensor2mask(mask)

        if mask_mapping_optional is not None:
            image = image[mask_mapping_optional]

        # Scale the mask to be a matching size if it isn't
        B, H, W, _ = image.shape
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:, 0, :, :]
        MB, _, _ = mask.shape

        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)

        # Prepare an empty RGBA result tensor matching the original image dimensions
        result = torch.zeros_like(image)

        for i in range(0, B):
            # Create an alpha mask for the current image
            alpha_mask = torch.zeros((H, W))
            alpha_mask[mask[i] > 0] = 1

            # Apply alpha mask to keep original coordinates but apply mask
            alpha_channel = image[i, :, :, 3] * alpha_mask # Apply mask to alpha channel
            result[i, :, :, 0:3] = image[i, :, :, 0:3] # Copy RGB channels directly
            result[i, :, :, 3] = alpha_channel # Apply new alpha channel

        # Convert back to original image type if necessary
        if C == 1:
            return (tensor2mask(result),)
        elif C == 3 and torch.min(result[:, :, :, 3]) == 1:
            return (tensor2rgb(result),)
        else:
            return (result,)