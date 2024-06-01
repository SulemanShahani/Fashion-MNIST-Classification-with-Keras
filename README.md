# Fashion MNIST Classification with Keras

This project demonstrates how to build, train, evaluate, and interpret a neural network for classifying images from the Fashion MNIST dataset using TensorFlow and Keras.

)

## Overview
Fashion MNIST is a dataset of Zalando's article images consisting of 70,000 grayscale images in 10 categories, with 7,000 images per category. The goal is to classify these images into their respective categories.

## Dataset
The dataset is split into:
- **Training set:** 60,000 images
- **Test set:** 10,000 images

Each image is 28x28 pixels and labeled with one of the following classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model Architecture
The model is a simple feedforward neural network with the following layers:
- Flatten layer to convert each 28x28 image into a 784-dimensional vector.
- Dense layer with 400 neurons and ReLU activation.
- Dense layer with 100 neurons and ReLU activation.
- Dense layer with 10 neurons and softmax activation for the output.

## Training
The model is compiled with the Adam optimizer and the sparse categorical crossentropy loss function. It is trained for 20 epochs with a batch size of 32 and a validation split of 5,000 samples from the training set.

## Evaluation
The model's performance is evaluated using accuracy metrics on both the training, validation, and test sets. Additionally, the following evaluation tools are used:
- **Classification Report:** Provides precision, recall, F1-score, and support for each class.
- **ROC Curve and AUC:** Used for multi-class evaluation to plot the ROC curve and calculate the AUC.
- **Precision-Recall Curve:** Shows the trade-off between precision and recall for different threshold values.
- **Error Analysis:** Visualizes misclassified examples to understand model weaknesses.
- **Learning Curves:** Plots training and validation accuracy/loss over epochs to diagnose overfitting or underfitting.

## Results
After training for 20 epochs, the model achieved:
- **Training accuracy:** 94.00%
- **Validation accuracy:** 89.84%
- **Test accuracy:** To be filled after running evaluation on the test set.

## Next Steps
1. **Early Stopping:** Implement early stopping to halt training when the validation loss stops improving.
2. **Learning Rate Adjustment:** Use learning rate schedules or reduce the learning rate on plateau.
3. **Error Analysis:** Further analyze misclassified examples to improve the model's performance.

## References
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)


## Usage
To run this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fashion-mnist-classification.git
    cd fashion-mnist-classification
    ```

2. **Install dependencies:**
    ```bash
    pip install tensorflow numpy pandas matplotlib scikit-learn seaborn
    ```

3. **Run the script:**
    ```bash
    python fashion_mnist_classification.py
    ```

Ensure you have Python and pip installed on your machine before running the commands above.

## Running the Code
- The script trains the model, evaluates it on the test set, and generates various evaluation metrics and visualizations.
- Modify the script as needed for further experimentation or improvements.

