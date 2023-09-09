# Oxford Pet Classification using AlexNet

This repository contains a Python script for image classification using the AlexNet architecture on the Oxford-IIIT Pet Dataset. It demonstrates how to load and preprocess the dataset, create a deep neural network model, train the model, and evaluate its performance.

## Dependencies

Before running the code, make sure you have the following dependencies installed:

- Python (>= 3.6)
- PyTorch (>= 1.0)
- torchvision
- Matplotlib
- NumPy
- SSL (for handling HTTPS)

You can install these dependencies using `pip`:

```bash
pip install torch torchvision matplotlib numpy
```

## Dataset

We use the Oxford-IIIT Pet Dataset, which contains images of 37 different breeds of cats and dogs. The dataset is divided into training, validation, and test sets. It can be downloaded and loaded using the `torchvision.datasets` module.

## Data Preprocessing

- The images are resized to (227x227) pixels.
- Pixel values are normalized to the range [-1, 1].

## Model Architecture

We use the AlexNet architecture for image classification. The architecture consists of five convolutional layers followed by three fully connected layers. It is implemented as a PyTorch `nn.Module` subclass.

## Training

- Cross-entropy loss is used as the loss function for multi-class classification.
- Stochastic Gradient Descent (SGD) is used as the optimizer with a learning rate of 0.001 and momentum of 0.9.

## GPU Support

The code automatically checks for the availability of a CUDA-compatible GPU. If a GPU is available, the code will run on the GPU; otherwise, it will fall back to CPU execution.

## Running the Code

You can run the code by executing the provided script. Make sure to adjust the batch size and other hyperparameters as needed. You can also modify the model architecture or training parameters as per your requirements.

## Results

The code evaluates the model's accuracy on the test set and prints the results.

## Author

Varun Kapuria

For any questions or issues, please contact varunkapuria@arizona.edu
