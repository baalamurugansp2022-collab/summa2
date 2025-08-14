# CELL 6: ADVANCED EVALUATION + RICH VISUALIZATIONS
#
# This script corresponds to the sixth cell of our Kaggle notebook.
# It provides a comprehensive suite for evaluating the models, calculating metrics,
# and generating rich visualizations and XAI (Explainable AI) outputs.

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm.notebook import tqdm
import cv2
import shap
from gradcam import GradCAM
from gradcam.utils import visualize_cam

# Import from previous cells
try:
    from cell_3_models import ADAS_MAML_RL_Net, load_benchmark_models
    # We would also need dataloaders from cell_2
except ImportError:
    print("Warning: Could not import from previous cells. Using placeholders.")
    from torch.utils.data import Dataset, DataLoader
    class ADAS_MAML_RL_Net(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = torch.nn.Linear(1,1)
        def forward(self, x): return { 'corrected_image': x, 'weather_logits': torch.randn(x.size(0), 5), 'segmentation_map': torch.randn(x.size(0), 10, x.size(2), x.size(3))}
    def load_benchmark_models(): return None, None

# --- 1. METRIC CALCULATION HELPERS ---

def calculate_miou(pred_mask, true_mask, num_classes=10):
    """Calculates mean Intersection over Union (mIoU) for segmentation."""
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    iou_per_class = []
    for cls in range(num_classes):
        intersection = np.sum((pred_mask == cls) & (true_mask == cls))
        union = np.sum((pred_mask == cls) | (true_mask == cls))
        if union == 0:
            iou = 1.0 # If no ground truth or prediction, it's a perfect match
        else:
            iou = intersection / union
        iou_per_class.append(iou)
    return np.mean(iou_per_class)

# --- 2. EVALUATION ORCHESTRATOR ---

class Evaluator:
    """
    Manages the full evaluation pipeline for the main model and benchmarks.
    """
    def __init__(self, main_model, yolo_model, deeplab_model, test_loader, device):
        self.main_model = main_model.to(device).eval()
        self.yolo_model = yolo_model
        self.deeplab_model = deeplab_model.to(device).eval() if deeplab_model else None
        self.test_loader = test_loader
        self.device = device
        self.results = {
            'true_seg': [], 'pred_seg_main': [], 'pred_seg_deeplab': [],
            'true_clear': [], 'pred_corrected': [],
            'true_weather': [], 'pred_weather': []
        }

    def run_inference(self):
        """Runs inference on the test set and stores predictions."""
        print("--- Running inference on test set... ---")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Assuming batch is a dict like in the training cell
                img = batch['image'].to(self.device)
                true_clear = batch['clear_image'].to(self.device)
                true_seg = batch['segmentation_mask'].to(self.device)
                true_weather = batch['weather_label'].to(self.device)

                # Main model predictions
                main_out = self.main_model(img)
                pred_corrected = main_out['corrected_image']
                pred_seg_main = torch.argmax(main_out['segmentation_map'], dim=1)
                pred_weather = torch.argmax(main_out['weather_logits'], dim=1)

                # DeepLabV3 predictions
                if self.deeplab_model:
                    deeplab_out = self.deeplab_model(img)['out']
                    pred_seg_deeplab = torch.argmax(deeplab_out, dim=1)
                    self.results['pred_seg_deeplab'].append(pred_seg_deeplab.cpu().numpy())

                # Store results
                self.results['true_seg'].append(true_seg.cpu().numpy())
                self.results['pred_seg_main'].append(pred_seg_main.cpu().numpy())
                self.results['true_clear'].append(true_clear.cpu().numpy())
                self.results['pred_corrected'].append(pred_corrected.cpu().numpy())
                self.results['true_weather'].append(true_weather.cpu().numpy())
                self.results['pred_weather'].append(pred_weather.cpu().numpy())

        # Concatenate results
        for key in self.results:
            self.results[key] = np.concatenate(self.results[key], axis=0)
        print("--- Inference complete. ---")

    def calculate_and_print_metrics(self):
        """Calculates and prints all the performance metrics."""
        print("\n--- Calculating Performance Metrics ---")

        # Dehazing/Correction Metrics
        psnr_vals = [psnr(t, p, data_range=1.0) for t, p in zip(self.results['true_clear'], self.results['pred_corrected'])]
        ssim_vals = [ssim(t, p, data_range=1.0, channel_axis=0) for t, p in zip(self.results['true_clear'], self.results['pred_corrected'])]
        print(f"Weather Correction -> PSNR: {np.mean(psnr_vals):.2f} | SSIM: {np.mean(ssim_vals):.4f}")

        # Segmentation Metrics
        miou_main = np.mean([calculate_miou(p, t) for p, t in zip(self.results['pred_seg_main'], self.results['true_seg'])])
        print(f"Segmentation (Main Model) -> mIoU: {miou_main:.4f}")
        if self.deeplab_model:
            miou_deeplab = np.mean([calculate_miou(p, t) for p, t in zip(self.results['pred_seg_deeplab'], self.results['true_seg'])])
            print(f"Segmentation (DeepLabV3) -> mIoU: {miou_deeplab:.4f}")

        # Weather Classification Metrics
        accuracy = np.mean(self.results['pred_weather'] == self.results['true_weather'])
        print(f"Weather Classification -> Accuracy: {accuracy * 100:.2f}%")

        # NOTE: Detection mAP is complex and requires specialized libraries (e.g., torchmetrics)
        # and ground truth bounding boxes, which are not in our current dataset.
        print("Detection Metrics -> Skipped (requires bounding box labels and mAP library).")

    def plot_all_visualizations(self, num_classes=5):
        """Generates and displays all the requested visualizations."""
        print("\n--- Generating Visualizations ---")

        # Confusion Matrix for Weather Classification
        cm = confusion_matrix(self.results['true_weather'], self.results['pred_weather'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.title('Weather Classification Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

# --- 3. XAI (EXPLAINABLE AI) FUNCTIONS ---

def generate_gradcam_visualizations(model, img_tensor, target_layer):
    """Generates and displays Grad-CAM visualizations."""
    print(f"--- Generating Grad-CAM for layer: {target_layer} ---")
    try:
        # We need the raw logits from the model for Grad-CAM
        model_output = model(img_tensor.unsqueeze(0))
        # Use segmentation map as the target for explanation
        saliency_map = model_output['segmentation_map']
        # GradCAM needs a single value per pixel, so we take the max across classes
        saliency_map = torch.max(saliency_map, dim=1)[0]

        gradcam = GradCAM(model, target_layer)
        mask, _ = gradcam(saliency_map)
        heatmap, result = visualize_cam(mask, img_tensor)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(heatmap)
        plt.title("Grad-CAM Heatmap")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title("Image with Grad-CAM Overlay")
        plt.show()
    except Exception as e:
        print(f"Could not generate Grad-CAM: {e}")

def generate_shap_explanations(model, background_data, test_image):
    """Generates and displays SHAP explanations."""
    print("--- Generating SHAP Explanations (this may take a moment) ---")
    # SHAP for vision models works by masking parts of the image
    masker = shap.maskers.Image("blur(128,128)", test_image.shape)

    # We need a function that takes masked images and returns a scalar output (e.g., a class logit)
    def shap_model_wrapper(x):
        # x is a numpy array of shape (num_masks, C, H, W)
        x_tensor = torch.tensor(x, device=model.device)
        with torch.no_grad():
            logits = model(x_tensor)['weather_logits']
        return logits.cpu().numpy()

    explainer = shap.KernelExplainer(shap_model_wrapper, background_data[:10]) # Use a few samples for background
    shap_values = explainer.shap_values(test_image.unsqueeze(0).numpy(), nsamples=50) # Use few samples for speed

    # Plot the SHAP values
    shap.image_plot(shap_values, test_image.permute(1,2,0).numpy())


# --- 4. MAIN EXECUTION (for Notebook Cell) ---

if __name__ == '__main__':
    print("--- Setting up and Verifying Evaluation Pipeline ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dummy components for demonstration
    print("Loading dummy model and data...")
    main_model = ADAS_MAML_RL_Net()
    yolo_model, deeplab_model = load_benchmark_models()

    class DummyEvalDataset(Dataset):
        def __len__(self): return 50
        def __getitem__(self, idx):
            return { 'image': torch.rand(3, 320, 480), 'clear_image': torch.rand(3, 320, 480), 'segmentation_mask': torch.randint(0, 10, (320, 480)), 'weather_label': torch.randint(0, 5, (1,)).squeeze() }

    test_loader = DataLoader(DummyEvalDataset(), batch_size=4)

    # 2. Instantiate and run the Evaluator
    evaluator = Evaluator(main_model, yolo_model, deeplab_model, test_loader, DEVICE)
    evaluator.run_inference()
    evaluator.calculate_and_print_metrics()
    evaluator.plot_all_visualizations()

    # 3. Demonstrate XAI on a single sample
    print("\n--- Demonstrating XAI on a single test sample ---")
    sample_data = next(iter(test_loader))
    sample_image = sample_data['image'][0].to(DEVICE)

    # Grad-CAM
    # The target layer name depends on the model architecture.
    # For our MobileViT, 'backbone.layer4' is a good choice.
    generate_gradcam_visualizations(main_model, sample_image, main_model.backbone.layer4)

    # SHAP (can be slow)
    # generate_shap_explanations(main_model, sample_data['image'].numpy(), sample_image.cpu())
    print("SHAP demonstration is commented out as it is computationally expensive.")

    print("\n--- Evaluation and XAI pipeline setup is complete and verified. ---")
