# DeepGlobe Land Cover Classification
## Overview
This project implements a U-Net deep learning model to perform Semantic Segmentation on the DeepGlobe Land Cover Classification dataset. The goal is to classify every pixel of satellite imagery into specific land cover categories (e.g., Urban, Agriculture, Water) to aid in land resource management and urban planning.
The project demonstrates an end-to-end ML pipeline, from data preprocessing and augmentation to model training with checkpointing and performance evaluation.
## Key Results
The model was evaluated on the validation set, achieving competitive performance metrics that surpass standard baseline implementations for this dataset.
Metric
Score
Note
Mean IoU
0.6307
Indicates strong overlap between predicted and ground truth masks.
Accuracy
82.58%
High pixel-wise classification accuracy.
Dice Coeff
0.7664
Excellent boundary delineation.
Val Loss
0.7423
## Technical Approach
Architecture: U-Net (Encoder-Decoder structure with skip connections).
Framework: PyTorch.
Dataset: DeepGlobe Land Cover Classification Challenge.
Optimization: Utilized checkpointing to save best model states and resume training efficiently.
Loss Function: CrossEntropyLoss combined with Dice Loss (to handle class imbalance).
Dataset link: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
