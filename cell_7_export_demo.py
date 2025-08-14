# CELL 7: MODEL EXPORT + INTERACTIVE DEMOS
#
# This script corresponds to the final cell of our Kaggle notebook.
# It provides functions to save the trained model in multiple formats
# and creates an interactive widget for a live demonstration.

import torch
import numpy as np
from PIL import Image
import io
import cv2

# ipywidgets are essential for creating the interactive demo in a notebook
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except ImportError:
    print("Warning: ipywidgets not found. Interactive demo will not be available.")
    widgets = None

# Import from previous cells
try:
    from cell_3_models import ADAS_MAML_RL_Net, load_benchmark_models
    from cell_2_data import data_transforms # For preprocessing
except ImportError:
    print("Warning: Could not import from previous cells. Using placeholders.")
    class ADAS_MAML_RL_Net(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = torch.nn.Linear(1,1)
        def forward(self, x): return { 'corrected_image': torch.clamp(x * 0.8, 0, 1), 'segmentation_map': torch.randn(x.size(0), 10, x.size(2), x.size(3)), 'detection_cls_logits': torch.randn(x.size(0), 80*9, 7, 7) }
    def load_benchmark_models(): return None, None
    from torchvision import transforms
    data_transforms = {'val': transforms.ToTensor()}


# --- 1. MODEL EXPORT FUNCTIONS ---

def export_models(model, dummy_input, base_filename="adas_model"):
    """
    Exports the trained model to standard PyTorch, TorchScript, and ONNX formats.

    Args:
        model (nn.Module): The trained model to export.
        dummy_input (torch.Tensor): A sample input tensor for tracing.
        base_filename (str): The base name for the output files.
    """
    print(f"--- Exporting model to multiple formats with base name: {base_filename} ---")
    model.eval() # Set model to evaluation mode

    # 1. Standard PyTorch format (.pth)
    # Saves the model's learned parameters.
    pth_path = f"{base_filename}.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"Successfully saved PyTorch state_dict to: {pth_path}")

    # 2. TorchScript format (.pt)
    # A JIT-compiled version for environments without Python.
    try:
        ts_path = f"{base_filename}.pt"
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(ts_path)
        print(f"Successfully saved TorchScript model to: {ts_path}")
    except Exception as e:
        print(f"Could not export to TorchScript: {e}")

    # 3. ONNX format (.onnx)
    # For cross-platform deployment and optimization.
    try:
        onnx_path = f"{base_filename}.onnx"
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=[f'output_{i}' for i in range(len(model(dummy_input)))],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}})
        print(f"Successfully saved ONNX model to: {onnx_path}")
    except Exception as e:
        print(f"Could not export to ONNX: {e}")

# --- 2. INTERACTIVE DEMO ---

class InteractiveDemo:
    """
    A class to create and manage the interactive demo widget.
    """
    def __init__(self, main_model, yolo_model, deeplab_model, device):
        if not widgets:
            print("ipywidgets is not installed. Cannot create interactive demo.")
            return

        self.main_model = main_model.to(device).eval()
        self.yolo_model = yolo_model
        self.deeplab_model = deeplab_model.to(device).eval() if deeplab_model else None
        self.device = device
        self.transform = data_transforms['val'] # Use validation transforms

        # Create widgets
        self.uploader = widgets.FileUpload(accept='image/*', description='Upload Image')
        self.run_button = widgets.Button(description='Run Analysis', button_style='success')
        self.output_widget = widgets.Output()

        self.run_button.on_click(self.on_run_button_clicked)

    def display(self):
        """Displays the widgets in the notebook."""
        if not widgets: return
        print("--- Interactive ADAS Pipeline Demo ---")
        print("Upload an image and click 'Run Analysis' to see the models in action.")
        display(widgets.VBox([self.uploader, self.run_button, self.output_widget]))

    def on_run_button_clicked(self, b):
        """Callback function to run inference when the button is clicked."""
        with self.output_widget:
            clear_output(wait=True) # Clear previous results
            if not self.uploader.value:
                print("Please upload an image first.")
                return

            print("Processing image...")
            # Get uploaded image data
            input_bytes = self.uploader.value[next(iter(self.uploader.value))]['content']
            img = Image.open(io.BytesIO(input_bytes)).convert('RGB')
            img_np = np.array(img)

            # Preprocess image
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                main_out = self.main_model(img_tensor)

            # --- Visualize Results ---
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("ADAS Pipeline Analysis", fontsize=20)

            # 1. Original vs. Corrected Image
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            corrected_img = main_out['corrected_image'].squeeze().permute(1, 2, 0).cpu().numpy()
            axes[0, 1].imshow(corrected_img)
            axes[0, 1].set_title('Weather-Corrected Image (Main Model)')
            axes[0, 1].axis('off')

            # 2. Segmentation
            pred_seg_map = torch.argmax(main_out['segmentation_map'], dim=1).squeeze().cpu().numpy()
            color_mask = np.zeros_like(img_np)
            for class_id in np.unique(pred_seg_map):
                if class_id != 0: color_mask[pred_seg_map == class_id] = np.random.randint(0, 255, 3)
            seg_overlay = cv2.addWeighted(img_np, 0.6, color_mask, 0.4, 0)
            axes[1, 0].imshow(seg_overlay)
            axes[1, 0].set_title('Segmentation (Main Model)')
            axes[1, 0].axis('off')

            # 3. Detection (YOLOv8)
            if self.yolo_model:
                yolo_results = self.yolo_model(img)
                yolo_plot = yolo_results[0].plot() # plot() returns a BGR numpy array
                axes[1, 1].imshow(cv2.cvtColor(yolo_plot, cv2.COLOR_BGR2RGB))
                axes[1, 1].set_title('Object Detection (YOLOv8)')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'YOLO Model not loaded', ha='center')
                axes[1, 1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

# --- 3. MAIN EXECUTION (for Notebook Cell) ---

if __name__ == '__main__':
    print("--- Setting up Model Export and Interactive Demo ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load a dummy model for demonstration
    print("Loading dummy model for export and demo setup...")
    main_model = ADAS_MAML_RL_Net().to(DEVICE)
    yolo_model, deeplab_model = load_benchmark_models()

    # 2. Demonstrate model export
    dummy_input = torch.randn(1, 3, 320, 480, device=DEVICE)
    export_models(main_model, dummy_input)

    # 3. Create and display the interactive demo
    # This part needs to be run in a Jupyter/Kaggle environment to work.
    print("\n--- To launch the demo, run the following in a notebook cell: ---")
    print("demo = InteractiveDemo(main_model, yolo_model, deeplab_model, DEVICE)")
    print("demo.display()")

    # In a script, we can't display the widget, but we can verify it instantiates
    try:
        demo = InteractiveDemo(main_model, yolo_model, deeplab_model, DEVICE)
        print("\nInteractiveDemo class instantiated successfully.")
    except Exception as e:
        print(f"Could not instantiate InteractiveDemo: {e}")

    print("\n--- Model export and demo setup is complete. ---")
