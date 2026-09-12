# Convolutional Neural Network (CNN) Image Classifier

> Originally developed as a university project, this repository demonstrates foundational Computer Vision concepts using pytorch. It showcases a custom data-loading pipeline, a from-scratch CNN architecture, and a command-line interface for both training and inference.

The model is built using a classic VGG-style architecture tailored for efficient spatial feature extraction on $128 \times 128$ images:

**Feature extractor:** a 3-block Convolutional pipeline (`3 -> 16 -> 32 -> 64` channels). Each block utilizes $3 \times 3$ kernels with padding, ReLU activations, and $2 \times 2$ Max Pooling to progressively downsample spatial dimensions while increasing depth.

**Classifier:** a fully connected feed-forward network. The flattened feature map is passed through a 256-unit dense layer, heavily regularized with a `Dropout(0.5)` layer to prevent overfitting on limited academic datasets, before outputting class logits.

**Data pipeline**

Implements a custom `torch.utils.data.Dataset` (`MainDataset`) to dynamically map images from a directory structure using a CSV index.

Applies standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) to stabilize gradient descent during training.

**Quick start**

```bash
# Ensure you have PyTorch, Pandas, and Pillow installed:
pip install torch pandas torchvision pillow

# Training the model
python main.py --train --epochs 15 --csv wonders_of_world_images.csv --data_dir .

# Running Inference
python main.py --predict test_image.jpg