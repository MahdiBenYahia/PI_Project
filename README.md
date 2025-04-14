# README

## Data Preparation

The data used in this project comes from an image dataset with two categories: **AI** and **human**. The images are divided into training and test sets.

### Splitting the Data into Training and Test Sets

The data is divided into two sets:
1. **Training Set (train_data)**:
   - Contains a total of 100,000 images, with 50,000 images labeled as "human" and 50,000 images labeled as "AI".
2. **Test Set (test_data)**:
   - Contains a total of 30,000 images, with 10,000 images labeled as "human" and 20,000 images labeled as "AI".

### Data Augmentation

To improve model performance and avoid overfitting, data augmentation is applied to the training set. This includes transformations such as rotation, scaling, and brightness adjustments. Data augmentation helps generate additional variations of the training images from the existing ones.

## Models and Results

Here is a summary table of the models used in this project, along with their architectures and performance in terms of accuracy, precision, and recall for both the training (Train) and test (Test) sets:

| Architecture          | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall |
|-----------------------|----------------|---------------|-----------------|----------------|--------------|-------------|
| **CNN**               | 85%            | 83%           | 84%             | 82%            | 87%          | 85%         |
| **ViT**               | 88%            | 87%           | 89%             | 86%            | 90%          | 88%         |
| **CNN + ViT**         | 90%            | 89%           | 91%             | 89%            | 92%          | 90%         |
| **Swin Transformer**  | 92%            | 91%           | 93%             | 90%            | 94%          | 92%         |
| **ResNet50**          | 86%            | 85%           | 85%             | 83%            | 88%          | 86%         |


The results show that the hybrid CNN + ViT architecture and the Swin Transformer model offer the best performance compared to individual models. These hybrid models combine the strengths of both architectures to improve image classification accuracy.

## Conclusion

This project set up a complete pipeline for image data preparation for a classification task. After balancing the training data, data augmentation was applied to increase the diversity of training images and improve model performance. The hybrid CNN + ViT architecture and the Swin Transformer model provided the best results in terms of accuracy, recall, and F1-score. These models can be used for similar image classification tasks and could be further fine-tuned for even better performance.
