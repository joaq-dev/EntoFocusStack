#!/bin/bash

# Install specific versions of PyTorch and TorchVision
pip install torch==2.1.0 torchvision==0.16.0 python==3.11

# Install other dependencies
pip install "numpy<2" pandas pillow opencv-python-headless natsort rawpy \
            lpips pytorch-msssim glob2 json5 subprocess32 \
            scikit-image argparse datetime

# Optional: add more dependencies if needed based on the image
echo "All dependencies installed successfully."
