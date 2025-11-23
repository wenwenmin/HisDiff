# HisDIff
## Overview
HisDiff is a novel deep learning framework designed to infer high resolution spatial gene expression profiles directly from H&E stained whole slide images. By integrating a hierarchical feature extraction module, which captures global context, neighborhood interactions, and spot specific details, with a powerful conditional diffusion framework, HisDiff effectively models the complex, one to many relationship between tissue morphology and transcriptional states. This generative approach overcomes the oversmoothing limitations of traditional regression methods, enabling the faithful recovery of biologically meaningful gene coexpression structures and functional pathways from low cost histological data.

![Overview.png](Overview.png)

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 3090)
- `pip install -r requiremnt.txt`

## Data
The datasets employed in this article can be downloaded via the dataset_download_hest1k.ipynb notebook included in this repository. This notebook provides scripts to acquire the Her2ST, cSCC, DLPFC, and PRAD datasets used for our evaluations.
### Data preprocessing
See [preprocess.ipynb](preprocess.ipynb) for the complete preprocessing pipeline

## Getting access
In our multimodal feature mapping extractor, the ViT architecture utilizes a self-pretrained model called UNI. You need to request access to the model weights from the Huggingface model page at:[https://huggingface.co/mahmoodlab/UNI](https://huggingface.co/mahmoodlab/UNI). It is worth noting that you need to apply for access to UNI login and replace it in the [preprocess.ipynb](preprocess.ipynb).

## Run HisDiff
run [train.py](train.py)

## Baselines
We have listed the sources of some representative baselines below, and we would like to express our gratitude to the authors of these baselines for their generous sharing.

- [iStar](https://github.com/daviddaiweizhang/istar) super-resolution gene expression from hierarchical histological features using a feedforward neural network. 
- [XFuse](https://github.com/ludvb/xfuse) integrates Spatial transcriptomics (ST) data and histology images using a deep generative model to infer super-resolution gene expression profiles. 
- [TESLA](https://github.com/jianhuupenn/TESLA) generates high-resolution gene expression profiles based on Euclidean distance metric, which considers the similarity in physical locations and histology image features between superpixels and measured spots.
- [STAGE](https://github.com/zhanglabtools/STAGE) to generate gene expression data for unmeasured spots or points from Spatial Transcriptomics with a spatial location-supervised Auto-encoder GEnerator by integrating spatial information and gene expression data. 

## Acknowledgements
Part of the code, such as the training framework based on pytorch lightning and the method for mask image in this repository is adapted from the [iStar](https://github.com/daviddaiweizhang/istar). And the Vision Transformer in this repository has been pre-trained by [UNI](https://github.com/mahmoodlab/UNI). We are grateful to the authors for their excellent work.

## Contact details
If you have any questions, please contact aixoneeee@gmailc.com.
