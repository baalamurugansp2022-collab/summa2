# CELL 1: ENVIRONMENT SETUP & VISUALIZATION STACK
#
# This script corresponds to the first cell of our Kaggle notebook.
# It handles all necessary installations, imports, and configurations.

# --- 1. INSTALL DEPENDENCIES ---
# In a Kaggle notebook, you would run the following commands in a code cell
# by prefixing them with ! (e.g., !pip install torch)

"""
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm
pip install transformers
pip install learn2learn
pip install higher
pip install gym
pip install stable-baselines3[extra]
pip install ultralytics
pip install opencv-python-headless
pip install matplotlib seaborn plotly
pip install shap
pip install grad-cam
pip install torchinfo
"""

# --- 2. IMPORTS ---
import os
import sys
import time
import random
from collections import defaultdict

# Core ML/DL Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import timm
from torchinfo import summary

# MAML Libraries
import learn2learn as l2l
import higher

# RL Libraries
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Computer Vision and Data Handling
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# XAI Libraries
import shap
from gradcam.utils import visualize_cam
from gradcam import GradCAM

# YOLOv8
from ultralytics import YOLO

# Other utilities
from tqdm.notebook import tqdm
import warnings

# --- 3. CONFIGURATIONS ---

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configure device (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {DEVICE} ---")
if torch.cuda.is_available():
    print(f"--- GPU: {torch.cuda.get_device_name(0)} ---")

# Plotting configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
px.defaults.template = "plotly_dark"

print("--- Environment setup complete. All libraries imported and configurations set. ---")

# This is a placeholder to show how a main execution block would look.
# In the notebook, the code above would just be run in a cell.
if __name__ == "__main__":
    print("\nThis script is intended to be used as the first cell in a Jupyter/Kaggle notebook.")
    print("It sets up the environment but does not perform any standalone actions.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Timm version: {timm.__version__}")
    print(f"Device: {DEVICE}")
