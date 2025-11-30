# BHL_GPU_Enjoyers

# Create a new Conda environment with Python 3.11
conda create -n myenv python=3.11

# Activate the environment
conda activate myenv

# Install PyTorch with CUDA 12.1 support
Windows:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Linux:
pip3 install torch torchvision

# Install Hugging Face Transformers, Datasets, and scikit-learn
pip install transformers datasets scikit-learn nltk streamlit codecarbon dotenv