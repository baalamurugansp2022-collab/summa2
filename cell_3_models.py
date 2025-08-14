# CELL 3: ADVANCED MODEL ARCHITECTURE + MAML SETUP
#
# This script corresponds to the third cell of our Kaggle notebook.
# It defines our primary, all-in-one model architecture (Model C) based on MobileViT,
# and loads the benchmark models (YOLOv8, DeepLabv3).

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from ultralytics import YOLO
from torchinfo import summary

# --- 1. HELPER MODULES ---

class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpConv(nn.Module):
    """Upsampling Block: ConvTranspose2d -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.up(x)))

# --- 2. THE ALL-IN-ONE MULTI-TASK MODEL (MODEL C) ---

class ADAS_MAML_RL_Net(nn.Module):
    """
    The main, multi-task model architecture based on the user's specification.
    Integrates a MobileViT backbone, MAML layers, RL network, and multiple heads
    for weather correction, segmentation, and detection.
    """
    def __init__(self, num_weather_classes=5, num_seg_classes=10, num_det_classes=80, num_anchors=9, num_rl_actions=4):
        super().__init__()

        # --- BACKBONE (MobileViT) ---
        # We use a pretrained MobileViT and extract features at different stages.
        # This allows for skip connections in the segmentation decoder.
        self.backbone = timm.create_model('mobilevit_s', pretrained=True, features_only=True)

        # --- MAML LAYERS ---
        # As per the spec, these layers are for fast adaptation.
        self.maml_conv1 = ConvBlock(320, 256, kernel_size=3, padding=1)
        self.maml_conv2 = ConvBlock(256, 256, kernel_size=3, padding=1)

        # --- FEATURE FUSION ---
        # This part is a placeholder for a more complex fusion strategy.
        # The spec mentions concat, but the inputs are not clearly defined.
        # We'll use the final feature map from the MAML layers as the primary fused feature.
        self.fusion_fc = nn.Linear(256, 512)
        self.fusion_dropout = nn.Dropout(0.5)

        # --- WEATHER CLASSIFICATION HEAD ---
        self.weather_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_weather_classes)
        )

        # --- WEATHER CORRECTION HEAD (DEHAZING/DE-WEATHERING) ---
        # A simple U-Net-like decoder to reconstruct a clear image.
        # It takes features from the MAML layers.
        self.deweather_up1 = UpConv(256, 160)
        self.deweather_up2 = UpConv(160, 96)
        self.deweather_up3 = UpConv(96, 64)
        self.deweather_final = nn.Conv2d(64, 3, kernel_size=1) # Output a 3-channel RGB image

        # --- SEGMENTATION HEAD ---
        # A decoder that uses skip connections from the MobileViT backbone.
        # Note: Channel sizes are matched to MobileViT-S feature maps.
        # MobileViT-S feature info:
        # 0: {'num_chs': 32, 'reduction': 2, 'module': 'conv1'}
        # 1: {'num_chs': 64, 'reduction': 4, 'module': 'layer1'}
        # 2: {'num_chs': 96, 'reduction': 8, 'module': 'layer2'}
        # 3: {'num_chs': 160, 'reduction': 16, 'module': 'layer3'}
        # 4: {'num_chs': 320, 'reduction': 32, 'module': 'layer4'}
        self.seg_up1 = UpConv(256, 160) # from maml output
        self.seg_conv1 = ConvBlock(160 + 160, 128) # Skip conn from backbone layer3

        self.seg_up2 = UpConv(128, 96)
        self.seg_conv2 = ConvBlock(96 + 96, 64) # Skip conn from backbone layer2

        self.seg_up3 = UpConv(64, 32)
        self.seg_conv3 = ConvBlock(32 + 64, 32) # Skip conn from backbone layer1

        self.seg_final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.seg_final = nn.Conv2d(16, num_seg_classes, kernel_size=1)

        # --- OBJECT DETECTION HEAD ---
        # Following the spec for a RetinaNet-like head structure.
        self.det_conv = nn.Sequential(*[ConvBlock(256, 256) for _ in range(4)])
        self.det_cls_head = nn.Conv2d(256, num_det_classes * num_anchors, kernel_size=3, padding=1)
        self.det_reg_head = nn.Conv2d(256, 4 * num_anchors, kernel_size=3, padding=1)

        # --- RL INTEGRATION HEAD ---
        # These layers will be called by the RL agent, not in the main forward pass.
        # The input size (517) is a placeholder: 512 from fused features + 5 for weather state.
        self.rl_state_encoder = nn.Linear(512 + num_weather_classes, 256)
        self.rl_policy_net = nn.Linear(256, 128)
        self.rl_action_head = nn.Linear(128, num_rl_actions)
        self.rl_value_head = nn.Linear(128, 1)

    def forward(self, x):
        # --- Backbone Feature Extraction ---
        features = self.backbone(x)
        f1, f2, f3, f4, f5 = features # f1 is after conv1, f5 is final layer4 output

        # --- MAML Adaptation ---
        # In a normal forward pass, this just acts as a regular conv block.
        # During MAML training, `higher` will manage the gradients.
        maml_features = self.maml_conv1(f5)
        maml_features = self.maml_conv2(maml_features)

        # --- Global Feature Vector for Classification/RL ---
        global_features = F.adaptive_avg_pool2d(maml_features, 1).flatten(1)
        fused = self.fusion_dropout(F.relu(self.fusion_fc(global_features)))

        # --- Head Outputs ---

        # Weather classification
        weather_logits = self.weather_head(fused)

        # Weather correction (dehazing)
        dw = self.deweather_up1(maml_features)
        dw = self.deweather_up2(dw)
        dw = self.deweather_up3(dw)
        # We need more upsampling to reach original size
        dw_final = F.interpolate(dw, size=x.shape[2:], mode='bilinear', align_corners=False)
        corrected_image = torch.sigmoid(self.deweather_final(dw_final)) # Sigmoid to keep output in [0, 1]

        # Segmentation
        seg = self.seg_up1(maml_features)
        seg = torch.cat([seg, f4], dim=1) # Skip connection from layer3
        seg = self.seg_conv1(seg)

        seg = self.seg_up2(seg)
        seg = torch.cat([seg, f3], dim=1) # Skip connection from layer2
        seg = self.seg_conv2(seg)

        seg = self.seg_up3(seg)
        seg = torch.cat([seg, f2], dim=1) # Skip connection from layer1
        seg = self.seg_conv3(seg)

        seg = self.seg_final_up(seg)
        seg_map = self.seg_final(seg)
        # Upsample to original image size
        seg_map = F.interpolate(seg_map, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Detection
        det_features = self.det_conv(maml_features)
        det_cls_logits = self.det_cls_head(det_features)
        det_bbox_preds = self.det_reg_head(det_features)

        # --- RL Forward Pass (to be called by agent) ---
        # The RL agent will call this method with the fused features and weather state.
        # This is separated for clarity.

        return {
            "corrected_image": corrected_image,
            "weather_logits": weather_logits,
            "segmentation_map": seg_map,
            "detection_cls_logits": det_cls_logits,
            "detection_bbox_preds": det_bbox_preds,
            "fused_features": fused
        }

    def forward_rl(self, fused_features, weather_state):
        """A separate forward pass for the RL agent."""
        rl_input = torch.cat([fused_features, weather_state], dim=1)
        state_repr = F.relu(self.rl_state_encoder(rl_input))
        policy_features = F.relu(self.rl_policy_net(state_repr))

        action_logits = self.rl_action_head(policy_features)
        state_value = self.rl_value_head(policy_features)

        return action_logits, state_value

# --- 3. FUNCTIONS TO LOAD BENCHMARK MODELS ---

def load_benchmark_models():
    """Loads pre-trained YOLOv8 and DeepLabV3 models."""
    print("Loading benchmark models...")
    # Load YOLOv8 for detection
    yolo_model = YOLO('yolov8n.pt')  # Using the nano version for speed
    print("YOLOv8 model loaded.")

    # Load DeepLabV3 for segmentation
    deeplab_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    deeplab_model.eval() # Set to evaluation mode
    print("DeepLabV3 model loaded.")

    return yolo_model, deeplab_model

# --- 4. MAIN EXECUTION (for Notebook Cell) ---

if __name__ == '__main__':
    print("--- Defining and Verifying Model Architectures ---")

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the main model
    print("\nInstantiating the All-in-One ADAS Model (Model C)...")
    model_c = ADAS_MAML_RL_Net().to(DEVICE)

    # Create a dummy input tensor
    # Using the dimensions from cell_2_data.py
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE = 320, 480, 4
    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)

    # --- Verify Model C Forward Pass ---
    print("\nVerifying forward pass of Model C...")
    try:
        outputs = model_c(dummy_input)
        print("Forward pass successful!")
        print("Output shapes:")
        for name, tensor in outputs.items():
            print(f"  - {name}: {tensor.shape}")
    except Exception as e:
        print(f"ERROR during forward pass: {e}")

    # --- Print Model C Summary ---
    print("\n--- ADAS_MAML_RL_Net (Model C) Summary ---")
    # The summary can be very long, so we'll set a reasonable depth
    summary(model_c, input_size=(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH), col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=4)

    # --- Load Benchmark Models ---
    print("\n--- Loading Benchmark Models (A & B) ---")
    model_a_yolo, model_b_deeplab = load_benchmark_models()
    model_b_deeplab.to(DEVICE)

    print("\n--- Model architecture definition and verification complete. ---")
