#!/bin/bash

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install specific versions of PyTorch and TorchVision
pip install torch==2.0.0 torchvision==0.15.0

# Install other dependencies
pip install numpy pandas pillow opencv-python-headless natsort rawpy \
            lpips pytorch-msssim glob2 logging json5 subprocess32 \
            scikit-image argparse datetime submitit

# Optional: add more dependencies if needed based on the image
echo "All dependencies installed successfully."
