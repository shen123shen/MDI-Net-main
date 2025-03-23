# MDI-Net-main

Currently, encoder-decoder structured networks are
 widely used in the field of medical image segmentation, with
 many researchers enhancing model performance by integrating
 multi-scale features through skip connections. However, previous
 studies have not delved deeply into the skip connection schemes,
 such as what kind of fusion methods to use and how to perform
 the fusion, leaving these questions without definitive answers.
 Therefore, this paper compares several advanced multi-scale
 feature fusion methods and proposes a more economical approach
 in terms of parameter count and computational cost, while
 achieving superior results. Based on this fusion approach, we have
 designed a plug-and-play multi-scale feature fusion module and
 applied it to several classic models, effectively improving their
 performance with minimal impact on the number of parameters
 and computational load. Furthermore, leveraging the proposed
 multi-scale feature fusion module, we have designed the MDI
Net network, which has been evaluated on five public datasets,
 including a skin lesion segmentation task. Compared with existing
 state-of-the-art (SOTA) methods, our model consistently out
performs the most advanced models, demonstrating exceptional
 segmentation capabilities, thus confirming that MDI-Net is a
 powerful solution for medical image segmentation.

# Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:

ISIC-2018 (dermoscopy, 2,594 images fortraining, 100 images for validation, and 1,000 images for testing)

Kvasir-SEG (gastrointestinal polyp, 600 images for training, 200images for validation, and 200 images for testing)

BUSI (breast ultrasound, 399 images for training.113 images for validation, and 118 images for testing)

CVC-ClinicDB (colorectal cancer, 367 images for training, 123images for validation, and 122 images for testing)
The dataset path may look like:

/The Dataset Path/
 ├── ISIC-2018/
  │ ├── Train_Folder/
   │ │ ├── img
   │ │ ├── labelcol
  │ ├── Val_Folder/
   │ │ ├── img
   │ │ ├── labelcol
  │ ├── Test_Folder/
   │ │ ├── img
   │ │ ├── labelcol

 # Usage
 Installation
 
 git clone git@github.com:shen123shen/MDI-Net-main
 MDI-Net-main.git
 conda create -n cfseg python=3.8
 conda activate cfseg
 conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
