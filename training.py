import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, RandAffined, RandFlipd, ScaleIntensityd, EnsureChannelFirstd, ToTensord, LambdaD
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle

class CTDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = Compose([
            LambdaD(keys=['ct', 'seg'], func=lambda x: x[np.newaxis, ...]),
            ScaleIntensityd(keys=['ct'], minv=-1000.0, maxv=1000.0),
            RandAffined(
                keys=['ct', 'seg'],
                prob=0.3,
                rotate_range=(0.05, 0.05, 0.05),
                translate_range=(5, 5, 2),
                scale_range=(0.08, 0.08, 0.08),
                padding_mode='border',
                mode=('bilinear', 'nearest')
            ),
            RandFlipd(
                keys=['ct', 'seg'],
                spatial_axis=0,
                prob=0.5
            ),
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
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

def pad_to_depth(x, target_depth):
    current_depth = x.shape[1]
    if current_depth < target_depth:
        pad_amount = target_depth - current_depth
        pad_tensor = torch.zeros(x.shape[0], pad_amount, *x.shape[2:], 
                               device=x.device, dtype=x.dtype)
        return torch.cat([x, pad_tensor], dim=1)
    return x

def custom_collate_fn(batch):
    max_depth = max(sample['ct'].shape[1] for sample in batch)
    return {
        'ct': torch.stack([pad_to_depth(sample['ct'], max_depth) for sample in batch]),
        'seg': torch.stack([pad_to_depth(sample['seg'], max_depth) for sample in batch])
    }

class MultiPassUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        self.encoder1 = self._block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._block(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.bottleneck = self._block(base_channels*2, base_channels*4)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        self.decoder2 = self._block(base_channels*4, base_channels*2)
        self.upconv1 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.decoder1 = self._block(base_channels*2, base_channels)
        
        self.conv = nn.Conv3d(base_channels, out_channels, 1)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, 3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
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
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

def train(processed_data_path, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data from pickle file
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize model and optimizer
    model = MultiPassUNet3D(in_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Create data loader
    train_loader = DataLoader(
        CTDataset(data),
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    best_dice = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        dice_total = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            ct = batch['ct'].to(device)
            seg = batch['seg'].to(device)
            
            # First pass
            blank_context = torch.zeros_like(ct)
            input_pass1 = torch.cat([ct, blank_context], dim=1)
            output_pass1 = model(input_pass1)
            
            # Second pass
            with torch.no_grad():
                context = (torch.sigmoid(output_pass1) > 0.5).float()
            input_pass2 = torch.cat([ct, context], dim=1)
            output_pass2 = model(input_pass2)
            
            # Loss calculation
            loss = F.binary_cross_entropy_with_logits(output_pass1, seg) + \
                   F.binary_cross_entropy_with_logits(output_pass2, seg)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                dice = dice_score(torch.sigmoid(output_pass2), seg)
                epoch_loss += loss.item()
                dice_total += dice
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = dice_total / len(train_loader)
        scheduler.step(avg_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, "best_model.pth")
            print(f"New best model saved with Dice: {best_dice:.4f}")

if __name__ == "__main__":
    processed_data_path = "processed_files/complete_dataset.pkl" 
    train(processed_data_path)