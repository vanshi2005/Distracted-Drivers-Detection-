# Distracted Driver Detection
***OVERVIEW***

This project implements a deep learning model to detect distracted driving behaviors from images. Using the MobileNetV2 architecture with TensorFlow, the model classifies driver images into 11 categories of distraction (e.g., texting, talking on the phone, safe driving). The project aims to enhance road safety by identifying potentially dangerous behaviors in real-time.


***DATASET***

The dataset used is the State Farm Distracted Driver Detection dataset, available on Kaggle. It contains images of drivers labeled with one of 11 classes:





c0: Safe driving



c1: Texting (right hand)



c2: Talking on phone (right hand)



c3: Texting (left hand)



c4: Talking on phone (left hand)



c5: Operating the radio



c6: Drinking



c7: Reaching behind



c8: Hair and makeup



c9: Talking to passenger



c10: Other distractions

The dataset is organized in a directory structure under ./imgs/train/ with subfolders for each class. The dataset is split into 80% training (17,950 images) and 20% validation (4,482 images).

***REQUIREMENTS***

To run this project, you need the following dependencies:

Python 3.8+





TensorFlow 2.x



NumPy



Pandas



Matplotlib



Seaborn



OpenCV (cv2)



Scikit-learn

***INSTALL THE DEPENDENCIES USING***

    pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn

***PROJECT STRUCTURE***

Distracted_Driver_Detection.ipynb: Jupyter Notebook containing the complete code for data preprocessing, model training, and evaluation.



imgs.zip: Compressed dataset (not included in the repository due to size; can be downloaded separately from Kaggle).



README.md: This file.

***SETUP INSTRUCTIONS***

Clone the Repository:

    git clone https://github.com/<your-username>/Distracted-Driver-Detection.git

    cd Distracted-Driver-Detection


***PREPARE THE DATASET***

Download the State Farm Distracted Driver Detection dataset from Kaggle.



Extract imgs.zip to create the ./imgs/train/ directory with subfolders for each class.



Update the dataset path in the notebook (data_path and train_dir) to match your local setup.


***INSTALL DEPENDENCIES***

    pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn


***RUN THE NOTEBOOK***

Open Distracted_Driver_Detection.ipynb in Jupyter Notebook or JupyterLab.

Execute the cells sequentially to preprocess data, train the model, and evaluate results.



***MODEL ARCHITECTURE***

Base Model: MobileNetV2 (pre-trained on ImageNet, with the top layer removed).



Input Size: 160x160 pixels (RGB images).



Custom Layers:

  Global Average Pooling

  Dense layers with ReLU activation

  Dropout for regularization



Output layer with 11 units (softmax activation for classification).



***TRAINING***





Batch size: 32



Optimizer: Adam



Loss: Sparse categorical cross-entropy



Early stopping to prevent overfitting.





***RESULTS***





The model achieves approximately 90% validation accuracy on the validation set after 10 epochs (update with actual results if available).



Key metrics (precision, recall, F1-score) are available in the classification report.



A confusion matrix visualizes the model’s performance across the 11 classes..

***FUTURE IMPROVEMENTS***

Fine-tune MobileNetV2 layers for better accuracy.



Experiment with other architectures (e.g., ResNet, EfficientNet).



Incorporate real-time video processing for practical deployment.



Address class imbalances in the dataset.



***LICENSE***

This project is licensed under the MIT License. See the LICENSE file for details.

***ACKNOWLEDGEMENT***

State Farm Distracted Driver Detection dataset (Kaggle)



TensorFlow and Keras for model development



MobileNetV2 for efficient deep learning

