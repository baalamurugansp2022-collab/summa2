# CELL 2: UNIFIED DATASETS & COMPREHENSIVE VISUALIZATION
#
# This script corresponds to the second cell of our Kaggle notebook.
# It defines the datasets, dataloaders, and visualizations for our three data sources.

import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

# --- 1. CONFIGURATION AND CONSTANTS ---

# Assumed Kaggle input directory structure
# The user needs to add these datasets to their Kaggle notebook
# The slugs are derived from the URLs provided
BASE_DIR = '../input/'
RESIDE_OTS_PATH = os.path.join(BASE_DIR, 'outdoor-training-set-ots-reside/RESIDE/OTS')
IDD_PATH = os.path.join(BASE_DIR, 'new-idd-dataset/IDD_Segmentation')
IDD_AW_PATH = os.path.join(BASE_DIR, 'idd-aw-adverse-weather-drive-scenes-segmentation/idd-aw')

# Model input parameters
IMG_WIDTH = 480
IMG_HEIGHT = 320
BATCH_SIZE = 8

# --- 2. DATASET CLASS FOR DEHAZING (RESIDE OTS) ---

class DehazingDataset(Dataset):
    """
    Dataset for image dehazing, loading hazy and clear image pairs.
    Assumes the RESIDE OTS structure with 'hazy' and 'clear' subfolders.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.hazy_files = sorted(glob.glob(os.path.join(self.root_dir, 'hazy', '*.png')))
        self.clear_files = sorted(glob.glob(os.path.join(self.root_dir, 'clear', '*.jpg'))) # .jpg in RESIDE

        # Ensure filenames match between hazy and clear folders
        self.mapping = {os.path.basename(f).split('_')[0]: f for f in self.hazy_files}
        self.file_list = [f for f in self.clear_files if os.path.basename(f).split('.')[0] in self.mapping]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        clear_img_path = self.file_list[idx]
        clear_img_name = os.path.basename(clear_img_path).split('.')[0]
        hazy_img_path = self.mapping[clear_img_name]

        try:
            clear_image = Image.open(clear_img_path).convert('RGB')
            hazy_image = Image.open(hazy_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {clear_img_path} or {hazy_img_path}")
            print(e)
            # Return dummy data or skip
            return self.__getitem__((idx + 1) % len(self))


        if self.transform:
            # Apply the same random seed for transformations to ensure consistency if needed
            seed = np.random.randint(2147483647)

            torch.manual_seed(seed)
            hazy_image = self.transform(hazy_image)

            torch.manual_seed(seed)
            clear_image = self.transform(clear_image)

        return hazy_image, clear_image

# --- 3. DATASET CLASS FOR SEGMENTATION (IDD & IDD-AW) ---

class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation, loading images and their masks.
    Combines data from both the standard IDD and the IDD-AW datasets.
    """
    def __init__(self, idd_dir, idd_aw_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.masks = []

        # Load from IDD
        idd_img_dir = os.path.join(idd_dir, 'leftImg8bit/train')
        idd_mask_dir = os.path.join(idd_dir, 'gtFine/train')
        for city in os.listdir(idd_img_dir):
            city_img_path = os.path.join(idd_img_dir, city)
            city_mask_path = os.path.join(idd_mask_dir, city)
            for f in os.listdir(city_img_path):
                img_path = os.path.join(city_img_path, f)
                mask_path = os.path.join(city_mask_path, f.replace('_leftImg8bit.png', '_gtFine_labelIds.png'))
                if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)

        # Load from IDD-AW
        idd_aw_img_dir = os.path.join(idd_aw_dir, 'rgb/images')
        idd_aw_mask_dir = os.path.join(idd_aw_dir, 'rgb/masks')
        for weather in os.listdir(idd_aw_img_dir):
            weather_img_path = os.path.join(idd_aw_img_dir, weather)
            weather_mask_path = os.path.join(idd_aw_mask_dir, weather)
            for f in os.listdir(weather_img_path):
                 img_path = os.path.join(weather_img_path, f)
                 mask_path = os.path.join(weather_mask_path, f)
                 if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Apply transformations
        # The same random seed is used to ensure image and mask are transformed identically
        seed = np.random.randint(2147483647)

        if self.transform:
            torch.manual_seed(seed)
            image = self.transform(image)

        if self.target_transform:
            torch.manual_seed(seed)
            mask = self.target_transform(mask)
            # Squeeze to remove channel dimension from mask
            mask = mask.squeeze(0).long()

        return image, mask

# --- 4. TRANSFORMATIONS ---

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

mask_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST),
    transforms.ToTensor()
])


# --- 5. VISUALIZATION FUNCTIONS ---

def show_dehazing_samples(dataset, num_samples=3):
    """Displays a few hazy/clear image pairs from the dehazing dataset."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 4))
    fig.suptitle("Dehazing Dataset Samples (Hazy vs. Clear)", fontsize=16)

    for i in range(num_samples):
        hazy, clear = dataset[i * len(dataset) // num_samples]

        # De-normalize for visualization
        hazy = hazy.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        clear = clear.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

        axes[i, 0].imshow(np.clip(hazy, 0, 1))
        axes[i, 0].set_title(f"Hazy Sample {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.clip(clear, 0, 1))
        axes[i, 1].set_title(f"Clear Sample {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def show_segmentation_samples(dataset, num_samples=3):
    """Displays a few image/mask pairs from the segmentation dataset."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    fig.suptitle("Segmentation Dataset Samples (Image vs. Mask vs. Overlay)", fontsize=16)

    for i in range(num_samples):
        img, mask = dataset[i * len(dataset) // num_samples]

        # De-normalize image for visualization
        img_vis = img.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = np.clip(img_vis, 0, 1)

        # Convert mask to a color map
        mask_vis = mask.numpy()
        # Create a color map (simple version, can be expanded for actual class colors)
        color_mask = np.zeros((*mask_vis.shape, 3), dtype=np.uint8)
        # Just for visualization, assign random colors to classes
        for class_id in np.unique(mask_vis):
            if class_id != 0:
                color_mask[mask_vis == class_id] = np.random.randint(0, 255, 3)

        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(color_mask)
        axes[i, 1].set_title(f"Mask {i+1}")
        axes[i, 1].axis('off')

        overlay = cv2.addWeighted((img_vis * 255).astype(np.uint8), 0.7, color_mask, 0.3, 0)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay {i+1}")
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 6. MAIN EXECUTION (for Notebook Cell) ---

if __name__ == "__main__":
    print("--- Initializing Datasets and DataLoaders ---")

    # NOTE: In a real Kaggle environment, you must check if the paths exist first.
    # This script assumes they do. Add error handling for a robust notebook.

    try:
        # Dehazing Dataset
        print("\nLoading Dehazing Dataset (RESIDE)...")
        dehazing_dataset = DehazingDataset(root_dir=RESIDE_OTS_PATH, transform=data_transforms['train'])
        print(f"Found {len(dehazing_dataset)} hazy/clear image pairs.")
        dehazing_loader = DataLoader(dehazing_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # Show samples
        show_dehazing_samples(dehazing_dataset)

    except FileNotFoundError:
        print(f"ERROR: RESIDE dataset not found at {RESIDE_OTS_PATH}. Please check the path.")
    except Exception as e:
        print(f"An error occurred while loading the RESIDE dataset: {e}")

    try:
        # Segmentation Dataset
        print("\nLoading Segmentation Datasets (IDD + IDD-AW)...")
        segmentation_dataset = SegmentationDataset(idd_dir=IDD_PATH, idd_aw_dir=IDD_AW_PATH,
                                                 transform=data_transforms['train'],
                                                 target_transform=mask_transforms)
        print(f"Found {len(segmentation_dataset)} images with segmentation masks.")
        segmentation_loader = DataLoader(segmentation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # Show samples
        show_segmentation_samples(segmentation_dataset)

    except FileNotFoundError:
        print(f"ERROR: IDD or IDD-AW dataset not found. Please check paths:")
        print(f"- {IDD_PATH}")
        print(f"- {IDD_AW_PATH}")
    except Exception as e:
        print(f"An error occurred while loading the Segmentation datasets: {e}")

    print("\n--- Data loading and visualization setup is complete. ---")
