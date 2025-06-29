import torch
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, ScaleIntensityd, LambdaD, ToTensord
)
from processing import process_files
import os

class CTDataset(torch.utils.data.Dataset):
    def __init__(self, processed_data):
        self.data = processed_data
        self.transform = Compose([
            LambdaD(keys=['ct', 'seg'], func=lambda x: x[np.newaxis, ...]),  # (1, D, H, W)
            ScaleIntensityd(keys=['ct'], minv=-1000.0, maxv=1000.0),
            ToTensord(keys=['ct', 'seg'], dtype=torch.float32)
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        data_dict = {
            'ct': sample['ct'].astype(np.float32),
            'seg': sample['seg'].astype(np.float32)
        }
        return self.transform(data_dict)

class MultiPassUNet3D(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        self.encoder1 = self._block(in_channels, base_channels)
        self.pool1 = torch.nn.MaxPool3d(2)
        self.encoder2 = self._block(base_channels, base_channels*2)
        self.pool2 = torch.nn.MaxPool3d(2)
        self.bottleneck = self._block(base_channels*2, base_channels*4)
        self.upconv2 = torch.nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        self.decoder2 = self._block(base_channels*4, base_channels*2)
        self.upconv1 = torch.nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.decoder1 = self._block(base_channels*2, base_channels)
        self.conv = torch.nn.Conv3d(base_channels, out_channels, 1)

    def _block(self, in_channels, features):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, features, 3, padding=1),
            torch.nn.BatchNorm3d(features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(features, features, 3, padding=1),
            torch.nn.BatchNorm3d(features),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

def visualize_prediction(ct, seg, pred, slice_idx=None):
    """Visualize a slice of the CT, ground truth, and prediction"""
    
    # Convert to numpy and get middle slice if not specified
    ct_np = ct.squeeze().cpu().numpy()
    seg_np = seg.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    
    if slice_idx is None:
        slice_idx = ct_np.shape[0] // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(ct_np[slice_idx], cmap='gray')
    plt.title('CT Slice')
    
    plt.subplot(132)
    plt.imshow(seg_np[slice_idx], cmap='gray')
    plt.title('Ground Truth')
    
    plt.subplot(133)
    plt.imshow(pred_np[slice_idx], cmap='gray')
    plt.title('Prediction')
    
    plt.show()

def run_inference():
    print("Starting inference...")
    
    # Check if processed data exists, if not run processing
    processed_data_path = "processed_files/complete_dataset.pkl"
    if not os.path.exists(processed_data_path):
        print("Processed data not found. Running processing step...")
        process_files()
    
    # Load the processed data
    print("Loading data...")
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = MultiPassUNet3D(in_channels=2).to(device)
    
    # Load model weights
    print("Loading model weights...")
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully")
    
    # Run inference
    print("Running inference...")
    all_scores = []
    
    dataset = CTDataset(data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch in tqdm(loader, desc="Processing samples"):
        ct = batch['ct'].to(device)
        seg = batch['seg'].to(device)
        
        # Print shapes for debugging
        print(f"\nSample shapes:")
        print(f"CT shape: {ct.shape}")
        print(f"Seg shape: {seg.shape}")
        
        with torch.no_grad():
            # First pass
            blank_context = torch.zeros_like(ct)
            input_pass1 = torch.cat([ct, blank_context], dim=1)
            output_pass1 = model(input_pass1)
            
            # Second pass
            context = (torch.sigmoid(output_pass1) > 0.5).float()
            input_pass2 = torch.cat([ct, context], dim=1)
            output_pass2 = model(input_pass2)
            
            # Calculate dice score
            pred = torch.sigmoid(output_pass2)
            
            # Normalize and flip segmentation
            seg = seg.float()
            
            # Print value ranges for debugging
            print(f"CT range: [{ct.min():.2f}, {ct.max():.2f}]")
            print(f"Seg dtype: {seg.dtype}, range: [{seg.min():.2f}, {seg.max():.2f}]")
            print(f"Pred dtype: {pred.dtype}, range: [{pred.min():.2f}, {pred.max():.2f}]")
            
            # Visualize to check alignment
            visualize_prediction(ct, seg, pred)
            
            print("Seg unique values:", torch.unique(seg))
            print("Pred unique values:", torch.unique(pred))
            
            plt.hist(seg.cpu().numpy().flatten(), bins=20, alpha=0.5, label='seg')
            plt.hist(pred.cpu().numpy().flatten(), bins=20, alpha=0.5, label='pred')
            plt.legend()
            plt.show()
            
            pred_bin = (pred > 0.5).float()
            print("Pred bin sum:", pred_bin.sum().item())
            
            score = dice_score(pred, seg)
            all_scores.append(score.item())
            print(f"Dice Score: {score.item():.4f}")
            print("Seg sum:", seg.sum().item())
            print("Pred sum:", pred.sum().item())
    
    # Print final results
    mean_score = np.mean(all_scores)
    print(f"\nFinal Results:")
    print(f"Mean Dice Score: {mean_score:.4f}")
    print(f"Number of samples processed: {len(all_scores)}")

if __name__ == "__main__":
    try:
        print("Script started")
        run_inference()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()