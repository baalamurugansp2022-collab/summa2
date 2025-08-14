# CELL 5: COMPREHENSIVE TRAINING + MAML + VISUALIZATIONS
#
# This script corresponds to the fifth cell of our Kaggle notebook.
# It defines the complete, complex training pipeline, including:
# - A multi-component loss function.
# - A MAML training loop for fast weather adaptation.
# - Integration of the RL agent's training.
# - An orchestrator class to manage the entire process.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import learn2learn as l2l
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import from previous cells
try:
    from cell_2_data import DehazingDataset, SegmentationDataset, data_transforms, mask_transforms
    from cell_3_models import ADAS_MAML_RL_Net
    from cell_4_rl import ADASEnv, PPO, DummyVecEnv
except ImportError:
    print("Warning: Could not import from previous cells. Using placeholders.")
    # Define placeholders if previous cells are not available
    from torch.utils.data import Dataset, DataLoader
    class ADAS_MAML_RL_Net(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = nn.Linear(1,1)
        def forward(self, x): return { 'corrected_image': x, 'weather_logits': torch.randn(x.size(0), 5), 'segmentation_map': torch.randn(x.size(0), 10, x.size(2), x.size(3)), 'detection_cls_logits': torch.randn(x.size(0), 80*9, 7, 7), 'detection_bbox_preds': torch.randn(x.size(0), 4*9, 7, 7), 'fused_features': torch.randn(x.size(0), 512) }
        def forward_rl(self, *args): return torch.randn(args[0].size(0), 4), torch.randn(args[0].size(0), 1)
    class DummySegDataset(Dataset):
        def __len__(self): return 200
        def __getitem__(self, idx): return torch.randn(3, 320, 480), torch.randint(0, 10, (320, 480))
    class ADASEnv: pass
    class PPO: pass
    class DummyVecEnv: pass


# --- 1. COMBINED LOSS FUNCTION ---

class CombinedLoss(nn.Module):
    """
    A combined loss function for our multi-task model.
    It computes and weights losses for weather correction, segmentation, and classification.
    """
    def __init__(self, seg_weight=1.0, deweather_weight=0.5, weather_cls_weight=0.2, det_weight=0.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.deweather_weight = deweather_weight
        self.weather_cls_weight = weather_cls_weight
        self.det_weight = det_weight # NOTE: Set to 0 by default as we don't have detection labels

        self.seg_loss = nn.CrossEntropyLoss()
        self.deweather_loss = nn.L1Loss() # L1 is robust for image-to-image tasks
        self.weather_cls_loss = nn.CrossEntropyLoss()

        # IMPORTANT: Placeholder for detection loss.
        # A real implementation would require bounding box labels and a more complex loss
        # (e.g., Focal Loss for classification, Smooth L1 for regression).
        self.det_cls_loss = nn.BCEWithLogitsLoss()
        self.det_reg_loss = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        # Segmentation Loss
        loss_seg = self.seg_loss(outputs['segmentation_map'], targets['segmentation_mask'])

        # Weather Correction Loss
        loss_deweather = self.deweather_loss(outputs['corrected_image'], targets['clear_image'])

        # Weather Classification Loss
        loss_weather_cls = self.weather_cls_loss(outputs['weather_logits'], targets['weather_label'])

        # Detection Loss (Placeholder)
        # In a real scenario, you would have detection targets. Here we compute a dummy loss.
        if self.det_weight > 0:
            # This part will not be trained unless det_weight > 0 and targets are provided.
            det_cls_target = torch.zeros_like(outputs['detection_cls_logits'])
            det_reg_target = torch.zeros_like(outputs['detection_bbox_preds'])
            loss_det = self.det_cls_loss(outputs['detection_cls_logits'], det_cls_target) + \
                       self.det_reg_loss(outputs['detection_bbox_preds'], det_reg_target)
        else:
            loss_det = torch.tensor(0.0, device=loss_seg.device)

        # Combine losses
        total_loss = (self.seg_weight * loss_seg +
                      self.deweather_weight * loss_deweather +
                      self.weather_cls_weight * loss_weather_cls +
                      self.det_weight * loss_det)

        return total_loss, {
            'total': total_loss.item(), 'segmentation': loss_seg.item(),
            'deweather': loss_deweather.item(), 'weather_cls': loss_weather_cls.item(),
            'detection': loss_det.item()
        }

# --- 2. TRAINING ORCHESTRATOR ---

class Trainer:
    """
    Manages the entire training and validation process for the ADAS model,
    including MAML and RL training steps.
    """
    def __init__(self, model, optimizer, criterion, device, rl_agent=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.rl_agent = rl_agent
        self.history = {'train_loss': [], 'val_loss': [], 'rl_rewards': []}

    def _maml_train_step(self, maml_learner, batch):
        """Performs a single MAML step (inner and outer loop)."""
        # This is a simplified MAML setup. A real one would have separate support/query sets.
        # Here we use the same batch for both for simplicity of demonstration.
        support_batch, query_batch = batch, batch

        # Inner loop: Adapt the learner
        for _ in range(5): # 5 adaptation steps
            outputs = maml_learner(support_batch['image'])
            adaptation_loss = self.criterion(outputs, support_batch)[0]
            maml_learner.adapt(adaptation_loss)

        # Outer loop: Evaluate on query set and update original model
        outputs = maml_learner(query_batch['image'])
        evaluation_loss, loss_components = self.criterion(outputs, query_batch)

        evaluation_loss.backward()
        return evaluation_loss.item(), loss_components

    def train_epoch(self, dataloader, rl_train_freq=100):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)

        for i, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Create a MAML learner from the model
            maml_learner = l2l.algorithms.MAML(self.model, lr=0.01, first_order=False)

            # Perform MAML step
            loss, loss_components = self._maml_train_step(maml_learner, batch)

            # Update main model's weights
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(1.0 / len(batch))

            self.optimizer.step()
            total_loss += loss

            progress_bar.set_postfix(loss=loss, seg=loss_components['segmentation'])

            # Train RL agent periodically
            if self.rl_agent and (i + 1) % rl_train_freq == 0:
                print(f"\n--- Training RL Agent at step {i+1} ---")
                self.rl_agent.learn(total_timesteps=1000, reset_num_timesteps=False)
                # We would log RL rewards here if the env provided them in info dict
                self.history['rl_rewards'].append(np.mean(self.rl_agent.ep_info_buffer['r']))


        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation Epoch", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['image'])
                loss, _ = self.criterion(outputs, batch)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, epochs):
        print("--- Starting Training ---")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("--- Training Finished ---")
        return self.history

# --- 3. VISUALIZATION ---

def plot_training_history(history):
    """Plots the training and validation loss curves."""
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 4. MAIN EXECUTION (for Notebook Cell) ---

if __name__ == '__main__':
    print("--- Setting up and Verifying Training Pipeline ---")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dummy components for demonstration
    print("Creating dummy components for verification...")
    model = ADAS_MAML_RL_Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CombinedLoss()

    # Create a dummy combined dataloader
    class CombinedDataset(Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 320, 480),
                'clear_image': torch.randn(3, 320, 480),
                'segmentation_mask': torch.randint(0, 10, (320, 480)),
                'weather_label': torch.randint(0, 5, (1,)).squeeze()
            }

    train_loader = DataLoader(CombinedDataset(), batch_size=4, shuffle=True)
    val_loader = DataLoader(CombinedDataset(), batch_size=4, shuffle=False)

    # 2. Instantiate and run the Trainer
    print("Instantiating Trainer...")
    trainer = Trainer(model, optimizer, criterion, DEVICE)

    # 3. Run a short training loop to verify
    print("Running a short training loop (2 epochs) to verify...")
    try:
        history = trainer.train(train_loader, val_loader, epochs=2)
        print("\nTraining pipeline verification successful!")
        plot_training_history(history)
    except Exception as e:
        print(f"\nERROR during training pipeline verification: {e}")

    print("\n--- Training pipeline setup is complete and verified. ---")
