from .clipseg import CLIPSegText, CLIPSegImage, GeneralSwitch, ColorMatchImage, TiledImage
"""
@author: nullquant
@title: CLIPSegOpt
@nickname: clipseg opt
@description: This repository contains an enhanced custom nodes to generate masks for image 
inpainting tasks based on text prompts or images by CLIP.
"""

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPSegText": CLIPSegText,
    "CLIPSegImage": CLIPSegImage,
    "ImpactSwitch": GeneralSwitch,
    "ColorMatchImage": ColorMatchImage,
    "TiledImage": TiledImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSegText": "CLIPSegText",
    "CLIPSegImage": "CLIPSegImage",
    "ImpactSwitch": "GeneralSwitch",
    "ColorMatchImage": "ColorMatchImage",
    "TiledImage": "TiledImage",
}