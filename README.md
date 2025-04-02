# 🐱 Dog-Cat Classification 🐶

A convolutional neural network (CNN) model built with TensorFlow/Keras to classify images of dogs and cats.

## 📋 Project Overview

This project implements a CNN to distinguish between images of dogs and cats. The model is trained on a dataset of 2000 images and tested on 400 images. Each image is 100x100 pixels with 3 color channels (RGB).

## 📊 Dataset

The dataset consists of:
- `input.csv`: Training images (2000 images, 100×100×3 pixels)
- `labels.csv`: Training labels (2000 labels, 0 for dog and 1 for cat)
- `input_test.csv`: Test images (400 images, 100×100×3 pixels)
- `labels_test.csv`: Test labels (400 labels, 0 for dog and 1 for cat)

**Note:** 💾 The dataset files are large (input.csv: 1.43GB, input_test.csv: 286MB) and are managed using Git LFS.

## 🧠 Model Architecture

The model uses a simple but effective CNN architecture:
- Input: 100×100×3 images
- First convolutional layer: 32 filters with 3×3 kernel, ReLU activation
- First max pooling layer: 2×2 pool size
- Second convolutional layer: 32 filters with 3×3 kernel, ReLU activation
- Second max pooling layer: 2×2 pool size
- Flatten layer
- First dense layer: 64 neurons, ReLU activation
- Output layer: 1 neuron, sigmoid activation (0 for dog, 1 for cat)

The model is compiled with:
- Loss function: Binary cross-entropy
- Optimizer: Adam
- Metrics: Accuracy

## 📈 Performance

After training for 5 epochs with a batch size of 64:
- Training accuracy: ~99.17% ✅
- Testing accuracy: ~69.00% 🤔

The difference between training and testing accuracy suggests some overfitting, which could be addressed in future improvements.

## 🛠️ Requirements

- Python 3.6+
- Jupyter Notebook
- TensorFlow 2.x
- NumPy
- Matplotlib

## 🚀 Usage

1. Clone the repository:
```
git clone https://github.com/Aromal004/Dog-Cat-Classification.git
cd Dog-Cat-Classification
```

2. Install dependencies:
```
pip install jupyter tensorflow numpy matplotlib
```

3. Launch Jupyter Notebook:
```
jupyter notebook
```

4. Open the notebook file:
```
DogOrCat.ipynb
```

5. Run all cells in the notebook to train and test the model

## 🔮 Future Improvements

- Add data augmentation to improve model generalization
- Implement dropout layers to reduce overfitting
- Try transfer learning with pre-trained models like VGG16 or ResNet
- Expand the dataset size for better training
- Fine-tune hyperparameters for improved performance

