# Custom Nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)


### This repository contains two custom nodes for ComfyUI that utilize the [CLIPSeg model](https://huggingface.co/docs/transformers/main/en/model_doc/clipseg) to generate masks for image inpainting tasks based on text and visual prompts.


This work is heavily based on https://github.com/biegert/ComfyUI-CLIPSeg by biegert, and its fork https://github.com/hoveychen/ComfyUI-CLIPSegPro by hoveychen.



### 1. CLIPSegText
The CLIPSegText node generates a binary mask for a given input image and text prompt.

**Inputs:**

- images: A torch.Tensor representing the input image or batch of images.
- text: A string representing the text prompt. Several prompts should be separated by comma.
- model_path: A local or net path to model. See below.
- blur: A float value to control the amount of Gaussian blur applied to the mask.
- threshold: A float value to control the threshold for creating the binary mask.
- dilation_factor: A float value to control the dilation of the binary mask.

**Outputs:**

- tensor_bw: A torch.Tensor representing the binary mask.
- image_out_hm: A torch.Tensor representing the heatmap overlay on the input image.
- image_out_bw: A torch.Tensor representing the binary mask overlay on the input image.


### 2. CLIPSegImage
The CLIPSegImage node generates a binary mask for a given input image and visual prompt.

**Inputs:**

- images: A torch.Tensor representing the input image or batch of images.
- visual_prompt: A torch.Tensor representing the image of visual prompt.
- model_path: A local or net path to model. See below.
- blur: A float value to control the amount of Gaussian blur applied to the mask.
- threshold: A float value to control the threshold for creating the binary mask.
- dilation_factor: A float value to control the dilation of the binary mask.

**Outputs:**

- tensor_bw: A torch.Tensor representing the binary mask.
- image_out_hm: A torch.Tensor representing the heatmap overlay on the input image.
- image_out_bw: A torch.Tensor representing the binary mask overlay on the input image.


## Path to model
To download model from Hugging Face use: "CIDAS/clipseg-rd64-refined".

If you want to use local install use path to installation, for example: "clipseg-rd64-refined/".
To download model from Hugging Face you will need git and git-lfs. Here is example of commands for Ubuntu:

```
sudo apt-get update
sudo apt-get install git git-lfs
git lfs install
git clone https://huggingface.co/CIDAS/clipseg-rd64-refined
cd clipseg-rd64-refined
git lfs pull
```


## Usage
Below is an example for the intended workflow. The [json file](workflow/workflow.json) for the example can be found inside the 'workflow' directory 
![](workflow/workflow.png?raw=true)

