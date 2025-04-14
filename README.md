# README

## Data Preparation

The data used in this project comes from an image dataset containing two categories: **AI** and **Human**. The images are split into training and testing sets.

### Train/Test Split

The dataset is divided into two main subsets:
1. **Training Set (train_data)**:
   - Contains 100,000 images: 50,000 labeled as "human" and 50,000 labeled as "AI".
2. **Testing Set (test_data)**:
   - Contains 30,000 images: 10,000 labeled as "human" and 20,000 labeled as "AI".

### Data Augmentation

To improve model performance and prevent overfitting, data augmentation is applied to the training set. This includes transformations such as rotation, scaling, and brightness adjustment. Augmentation increases the diversity of training examples by generating additional image variations from existing samples.

## Models and Results

The table below summarizes the models used in this project along with their architectures and performance metrics in terms of accuracy, precision, and recall for both training and testing:

| Architecture          | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall |
|-----------------------|----------------|---------------|-----------------|----------------|--------------|-------------|
| **CNN**               | 85%            | 83%           | 84%             | 82%            | 87%          | 85%         |
| **ViT**               | 88%            | 87%           | 89%             | 86%            | 90%          | 88%         |
| **CNN + ViT**         | 90%            | 89%           | 91%             | 89%            | 92%          | 90%         |
| **Swin Transformer**  | 92%            | 91%           | 93%             | 90%            | 94%          | 92%         |
| **ResNet50**          | 86%            | 85%           | 85%             | 83%            | 88%          | 86%         |

The results show that the **Swin Transformer** model achieved the best overall performance across all evaluation metrics, outperforming both individual and hybrid architectures. Its strong accuracy, precision, and recall make it particularly suitable for the task of distinguishing real art from AI-generated images.

## Conclusion

This project implemented a full pipeline for preparing image data for classification. After balancing the training data, data augmentation was applied to increase training diversity and boost performance. Among all models tested, the **Swin Transformer** demonstrated the best results in terms of accuracy, precision, and recall. It is a strong candidate for similar image classification tasks and can be further fine-tuned for even better performance.
