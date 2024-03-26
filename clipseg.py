from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import torch
import numpy as np

import cv2
import matplotlib.cm as cm
from PIL import Image

from scipy.ndimage import gaussian_filter

from typing import Tuple

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")


class CLIPSegText:

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {"required":
                    {
                        "images": ("IMAGE",),
                        "text": ("STRING", {"multiline": True}),
                        "model_path": ("STRING", {"multiline": False, "default": "CIDAS/clipseg-rd64-refined"}),
                     },
                "optional":
                    {
                        "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 7}),
                        "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.05, "default": 0.4}),
                        "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 4}),
                    }
                }

    CATEGORY = "image"
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("Mask","Heatmap Mask", "BW Mask")

    FUNCTION = "segment_image"

    def segment_image(self, images: torch.Tensor, text: str, model_path: str, blur: float, threshold: float, dilation_factor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a segmentation mask from an image and a text prompt using CLIPSeg.

        Args:
            images (torch.Tensor): The image to segment.
            text (str): The text prompt to use for segmentation, combine masks using commas.
            model_path (str): A local or net path to model.
            blur (float): How much to blur the segmentation mask.
            threshold (float): The threshold to use for binarizing the segmentation mask.
            dilation_factor (int): How much to dilate the segmentation mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The segmentation mask, the heatmap mask image, and the binarized mask image.
        """

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        processor = CLIPSegProcessor.from_pretrained(model_path)
        model = CLIPSegForImageSegmentation.from_pretrained(model_path)
        model.to(device)

        tensor_bws = []
        image_out_heatmaps = []
        image_out_binaries = []

        for image in images:
            image_pil, image_np = image_from_tensor(image)
            prompt = text.split(',')
            
            input_prc = processor(text=prompt, images=[image_pil] * len(prompt), padding=True, return_tensors="pt")
            input_prc = input_prc.to(device)

            # Predict the segmentation mask
            with torch.no_grad():
                model_outputs = model(**input_prc)
            
            outputs_cpu = model_outputs[0].to('cpu')
            if len(outputs_cpu.size()) < 3:
                outputs = outputs_cpu.unsqueeze(0)
            else:
                outputs = outputs_cpu

            image_out_heatmap, image_out_binary = masks_from_outputs(outputs, image_np, blur, threshold, dilation_factor)
            
            tensor_bws.append(image_out_binary[:, :, :, 0].squeeze(0))
            image_out_heatmaps.append(image_out_heatmap.squeeze(0))
            image_out_binaries.append(image_out_binary.squeeze(0))            

        return torch.stack(tensor_bws), torch.stack(image_out_heatmaps), torch.stack(image_out_binaries)


class CLIPSegImage:

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {"required":
                    {
                        "images": ("IMAGE",),
                        "visual_prompt": ("IMAGE",),
                        "model_path": ("STRING", {"multiline": False, "default": "CIDAS/clipseg-rd64-refined"}),
                     },
                "optional":
                    {
                        "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 7}),
                        "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.05, "default": 0.4}),
                        "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 4}),
                    }
                }

    CATEGORY = "image"
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("Mask","Heatmap Mask", "BW Mask")

    FUNCTION = "segment_image"

    def segment_image(self, images: torch.Tensor, visual_prompt: torch.Tensor, model_path: str, blur: float, threshold: float, dilation_factor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a segmentation mask from an image and a visual prompt using CLIPSeg.

        Args:
            images (torch.Tensor): The image to segment.
            image_prompt (torch.Tensor): The visual prompt to use for segmentation.
            blur (float): How much to blur the segmentation mask.
            threshold (float): The threshold to use for binarizing the segmentation mask.
            dilation_factor (int): How much to dilate the segmentation mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The segmentation mask, the heatmap mask image, and the binarized mask image.
        """

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        processor = CLIPSegProcessor.from_pretrained(model_path)
        model = CLIPSegForImageSegmentation.from_pretrained(model_path)
        model.to(device)

        tensor_bws = []
        image_out_heatmaps = []
        image_out_binaries = []

        for image in images:
            image_pil, image_np = image_from_tensor(image)
            vp_pil , _ = image_from_tensor(visual_prompt.squeeze(0))
            
            encoded_image = processor(images=[image_pil], return_tensors="pt")
            encoded_prompt = processor(images=[vp_pil], return_tensors="pt")
            encoded_image = encoded_image.to(device)
            encoded_prompt = encoded_prompt.to(device)            

            # Predict the segmentation mask
            with torch.no_grad():
                model_outputs = model(**encoded_image, conditional_pixel_values=encoded_prompt.pixel_values)

            outputs_cpu = model_outputs[0].to('cpu')
            if len(outputs_cpu.size()) < 3:
                outputs = outputs_cpu.unsqueeze(0)
            else:
                outputs = outputs_cpu

            image_out_heatmap, image_out_binary = masks_from_outputs(outputs, image_np, blur, threshold, dilation_factor)

            tensor_bws.append(image_out_binary[:, :, :, 0].squeeze(0))
            image_out_heatmaps.append(image_out_heatmap.squeeze(0))
            image_out_binaries.append(image_out_binary.squeeze(0))            

        return torch.stack(tensor_bws), torch.stack(image_out_heatmaps), torch.stack(image_out_binaries)


def image_from_tensor(t: torch.Tensor):
    # Convert the Tensor to a PIL image
    image_np = t.numpy() 
    # Convert the numpy array back to the original range (0-255) and data type (uint8)
    image_np = (image_np * 255).astype(np.uint8)
    # Create a PIL image from the numpy array
    return (Image.fromarray(image_np, mode="RGB"), image_np)

def dilate_mask(mask: torch.Tensor, dilation_factor: float) -> torch.Tensor:
    """Dilate a mask using a square kernel with a given dilation factor."""
    kernel_size = int(dilation_factor * 2) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask.numpy(), kernel, iterations=1)
    return torch.from_numpy(mask_dilated)

def apply_colormap(mask: torch.Tensor, colormap) -> np.ndarray:
    """Apply a colormap to a tensor and convert it to a numpy array."""
    colored_mask = colormap(mask.numpy())[:, :, :3]
    return (colored_mask * 255).astype(np.uint8)

def resize_image(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    """Resize an image to the given dimensions using linear interpolation."""
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

def overlay_image(background: np.ndarray, foreground: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay the foreground image onto the background with a given opacity (alpha)."""
    return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]

def masks_from_outputs(outputs: torch.Tensor, image_np: np.ndarray, blur: float, threshold: float, dilation_factor: int) -> Tuple[torch.Tensor, torch.Tensor]:
    masks = []
    for output in outputs:
        tensor = torch.sigmoid(output) # get the mask

        # Apply a threshold to the original tensor to cut off low values
        tensor_thresholded = torch.where(tensor > threshold, torch.tensor(1, dtype=torch.float), torch.tensor(0, dtype=torch.float))
        masks.append(tensor_thresholded)

    masks = torch.stack(masks).max(dim=0)[0]

    # Apply Gaussian blur to the thresholded tensor
    sigma = blur
    tensor_smoothed = gaussian_filter(masks.numpy(), sigma=sigma)
    tensor_smoothed = torch.from_numpy(tensor_smoothed)

    # Normalize the smoothed tensor to [0, 1]
    mask_normalized = (tensor_smoothed - tensor_smoothed.min()) / (tensor_smoothed.max() - tensor_smoothed.min())

    # Dilate the normalized mask
    mask_dilated = dilate_mask(mask_normalized, dilation_factor)

    # Convert the mask to a heatmap and a binary mask
    heatmap = apply_colormap(mask_dilated, cm.viridis)
    binary_mask = apply_colormap(mask_dilated, cm.Greys_r)

    # Resize masks
    dimensions = (image_np.shape[1], image_np.shape[0])
    heatmap_resized = resize_image(heatmap, dimensions)
    binary_mask_resized = resize_image(binary_mask, dimensions)

    # Overlay the heatmap and binary mask on the original image
    alpha_heatmap, alpha_binary = 0.5, 1
    overlay_heatmap = overlay_image(image_np, heatmap_resized, alpha_heatmap)
    overlay_binary = overlay_image(image_np, binary_mask_resized, alpha_binary)

    # Convert the numpy arrays to tensors
    image_out_heatmap = numpy_to_tensor(overlay_heatmap)
    image_out_binary = numpy_to_tensor(overlay_binary)

    return (image_out_heatmap, image_out_binary)
