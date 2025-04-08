This project fine-tunes Stable Diffusion using LoRA (Low-Rank Adaptation) to generate oil painting style images based on a small custom dataset of reference paintings. It loads a pretrained Stable Diffusion model and injects lightweight LoRA adapters into key attention layers of the U-Net backbone, specifically targeting modules like to_q, to_k, to_v, and to_out.

The training process uses images as input, paired with a consistent style prompt such as "oil painting of Canadian nature." Only the LoRA adapters are trained, while the base model weights remain frozen, which makes training faster and more memory efficient.

Once training is complete, the model can generate stylized images in two ways:

Text-to-image, by generating new images in the learned style using a prompt

Image-to-image, by applying the learned painting style to an existing photo while preserving its overall content and structure

This approach is well suited for learning custom artistic styles with minimal data.
