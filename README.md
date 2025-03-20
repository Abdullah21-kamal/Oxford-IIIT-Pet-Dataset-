# Oxford-IIIT-Pet-Dataset-
## Deep Learning Project with Fine-Tuning & Data Augmentation
Project Overview
This project implements deep learning-based image classification using MobileNetV2 on the Oxford-IIIT Pet Dataset. The goal is to accurately classify 37 pet breeds by leveraging transfer learning, fine-tuning, and data augmentation.

## The project consists of two primary implementations:
- Fine-Tuning MobileNetV2 – Training the model on the pet dataset after freezing pre-trained layers.
- Data Augmentation for MobileNetV2 – Applying transformations (rotation, flipping, zooming, etc.) to improve model generalization.

Dataset Information
The Oxford-IIIT Pet Dataset consists of 37 different breeds of cats and dogs. It is used to train and evaluate the model's performance.

## Key Characteristics:

- Total Images: ~7,400
- Classes: 37 (various pet breeds)
- Challenges:
- Class imbalance
- Variations in pose, lighting, and background clutter
- High similarity among some breeds

## Project Structure
### Oxford-IIIT-Pet-Dataset/
- │── .ipynb_checkpoints/       # Jupyter Notebook checkpoints (auto-generated)
- │── data/                     # Dataset directory
- │── models/                   # Trained model files (if applicable)
- │── notebooks/                # Jupyter Notebooks for training and evaluation
- │   ├── MobileNetV2_FineTuning.ipynb
- │   ├── AugmentedDataForMobileNetV2_FineTuning.ipynb
- │── AugmetedDataForMobileNetV2_FineTuning.pdf  # Report for Augmented Model
- │── MobileNetV2_FineTuning.pdf # Report for Fine-Tuned Model
- │── README.md                 # Project documentation
- │── requirements.txt          # Dependencies list


## Installation & Setup
1. Clone the Repository

- git clone https://github.com/Abdullah21-kamal/Oxford-IIIT-Pet-Dataset.git
- cd Oxford-IIIT-Pet-Dataset

2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Download the Dataset
The dataset can be manually downloaded from Oxford-IIIT Pet Dataset -> https://www.robots.ox.ac.uk/~vgg/data/pets/
Or using TensorFlow Datasets:
import tensorflow_datasets as tfds
dataset, info = tfds.load("oxford_iiit_pet", with_info=True)

## Running the Notebooks
The project contains two Jupyter Notebooks that can be run interactively:

- 1. Fine-Tuning MobileNetV2 – MobileNetV2_FineTuning.ipynb
- 2. Data Augmentation + Fine-Tuning – AugmentedDataForMobileNetV2_FineTuning.ipynb

## To run the notebooks, use:
- jupyter notebook
- Then open the required notebook and execute the cells.

## Model Training & Evaluation
- Training Strategy
- MobileNetV2 pre-trained on ImageNet is used as the base model.
- Fine-tuning is applied to improve feature extraction.
- Data augmentation is used to reduce overfitting.
- Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix Analysis
## Results
- Model Version	vc Test Accuracy
- MobileNetV2 (Baseline)	86.64%
- MobileNetV2 + Fine-Tuning	87.27%
- MobileNetV2 + Data Augmentation	89.0%

## Fine-tuning and data augmentation significantly improved classification accuracy

## Predictions & Visualization
- The trained model was tested on new images to verify accuracy.
- Correct & misclassified images were visualized and analyzed.
- Confusion matrix analysis identified frequently misclassified breeds.

## Unit Testing
To ensure code reliability, unit tests can be written for data processing and model functions.

 To implement unit tests, create a tests/ directory and use pytest or unittest. 
## To be added.......

- Future Work
- 🔹 Improve data augmentation techniques (GANs, synthetic images)
- 🔹 Experiment with EfficientNet and Vision Transformers (ViTs)
- 🔹 Deploy as a web-based classification tool
- 🔹 Optimize for mobile devices using TensorFlow Lite

## Contributors
### Abdullah Kamal
- abdullah202kamal@gmail.com
- t-abdullahkamal@zewailcity.edu.eg
- abdullahkamal@aucegypt.edu
- Zewail City / AUC
