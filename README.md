# white blood cells classification model architecture 

White blood cell (WBC) classification is a crucial task in medical diagnosis, as it helps in identifying various types of WBCs and their abnormalities. In this research work, we propose a novel deep learning-based approach for WBC classification using a customized convolutional neural network (CNN) architecture.




## Over view of Model 


model is composed of four major components:

#1.Base Model(back bone) - MobilNet

#2.SK block - customized block 

#3.multifusion block - customized 

#4 Channel Attention Block (CAB) - customized 


    



## base model The model uses MobileNetV2 as the backbone architecture with pre-trained ImageNet weights
Input image size: 112x112x3
Three feature maps are extracted from different layers of MobileNetV2:

Feature1: from 'block_3_expand'

Feature2: from 'block_6_expand'

Feature3: from 'block_13_expand'

which are used to extract  features from the image. 



## SK block
This block implements selective kernel attention mechanism:


Contains two parallel branches:

Branch1: Uses 3×3 convolutions for local feature extraction

Branch2: Uses dilated 5×5 convolutions (dilation rates 3 and 5) for larger receptive field

Features dynamic selection between different kernel sizes
Helps in capturing multi-scale information
Uses attention mechanism to adaptively adjust the importance of different receptive fields

## model design of block 

## Context Attention Block 
This block enhances feature representation by:


Using depthwise convolutions with different kernel sizes (5×5 and 7×7)

Implementing channel attention mechanism
Incorporating skip connections for better gradient flow
Using GELU activation for non-linearity.

Purpose: Captures contextual information and enhances important features while suppressing irrelevant ones

## model design of block 


## Model pipeline 
1.Feature Extraction


Extracts features from three different levels of MobileNetV2
Applies spatial scaling (MaxPooling2D/UpSampling2D) to align feature dimensions


2.Feature Enhancement

Each feature map passes through the CAB Block
Features are processed through SK Block for multi-scale feature extraction
Outputs are combined  using concatenated operation


3.Classification Head

-Global average pooling

-Dense layer with 256 units and ReLU activation

-Final dense layer with 5 units (number of classes) and softmax  activation

Training Details:

Uses data augmentation (rotation, zoom, flips)

Adam optimizer

Categorical crossentropy losses
Batch size: 8, 16 ,32 
Model checkpointing for best validation accuracy



This architecture is designed to effectively capture both fine-grained details and global context necessary for accurate white blood cell classification.
## Results 


## validation accuracy 

 Trained model on PBC datasets which contain five thousands     images along with different batch sizes -8,16,32 . 
 we had got excellent results on 8 and with the training epoch -56 .


 ## ROC curve :

 i have plot  the ROC curve for our above results 




 ## confusion matrix :
 i have plot the matrix for  my results 


 ## Training validation  accuracy plot :
  we have found very good results for it . 



## Training validation loss plots : 




## References :
i have  taken the reference  of these papers 
to develop my model :
https://ieeexplore.ieee.org/document/10274670
 
And some other papers 



