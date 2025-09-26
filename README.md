# Plant Classification with CNN

This repository contains a complete deep learning pipeline for plant image classification using Convolutional Neural Networks (CNNs). The goal is to classify plant images into different categories while providing insights into the model’s decision-making process.

# Introduction

The project includes the following steps:

- **Dataset Preparation**: Cleaned and validated the dataset, removed corrupted and duplicate images, and created metadata CSV files.  
- **Preprocessing & Standardization**: Converted all images to RGB, resized them consistently, and generated label mappings.  
- **Data Loading & Augmentation**: Applied transformations (flips, rotations, crops, color jitter) for training; used PyTorch DataLoader for batching.  
- **Model Development**: Built two CNN models:
  - `SimpleCNN`: baseline model with 3 convolutional layers, dropout, and two fully connected layers.  
  - `AdvancedCNN`: deeper model with 4 convolutional layers, Batch Normalization, dropout, and weight decay.  
- **Training & Evaluation**: Trained using Adam optimizer and CrossEntropyLoss; tracked train/validation loss and accuracy. Evaluated test set performance with Accuracy, F1 Score, Precision, Recall, and Specificity.  
- **Explainability**: Implemented Grad-CAM to visualize image regions influencing model predictions.

End-to-end workflow: **Data cleaning → Preprocessing → CNN training → Evaluation → Explainability.**

# Metrics

**AdvancedCNN Results (12 epochs):**

| Metric      | Value  |
|------------|--------|
| Accuracy   | 0.455  |
| F1 Score  | 0.440  |
| Precision | 0.471  |
| Recall    | 0.455  |
| Specificity | 0.981 |

- **Interpretation**:
  - The model correctly predicts roughly half of the test images.  
  - F1 Score indicates moderate balance between precision and recall.  
  - High specificity (~98%) shows that the model rarely misclassifies other classes as a given class.  
  - Overall, the model has learned meaningful patterns, but there is room for improvement through data augmentation, deeper architectures, or hyperparameter tuning.

# Additional Notes

- Attempted to use `pytorch_grad_cam` for Grad-CAM/Eigen-CAM visualizations but encountered installation issues

- **Alternative solution**: Implemented a manual Grad-CAM approximation using forward hooks and backpropagation. The resulting heatmaps successfully highlight image regions contributing to predictions, providing interpretable visualizations without relying on external libraries.

# Conclusion and Future Work

- The AdvancedCNN improves over simpler CNNs in accuracy, learning stability, and interpretability.  
- Future improvements could include:
  - More aggressive data augmentation.  
  - Using deeper or pre-trained CNN architectures.  
  - Hyperparameter tuning (learning rate, batch size, dropout).  
  - Adding deployment scripts or interactive UI for end-to-end use.

# Links

- https://www.kaggle.com/code/mayayagan/notebookb2e2ee72e3

- https://www.kaggle.com/datasets/marquis03/plants-classification

