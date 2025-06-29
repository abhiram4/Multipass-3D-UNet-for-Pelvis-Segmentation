import os
import subprocess
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
from pathlib import Path

def get_patient_folders(root_path):
    return [
        os.path.join(root_path, d) 
        for d in os.listdir(root_path) 
        if os.path.isdir(os.path.join(root_path, d)) 
        and "im_1" in os.listdir(os.path.join(root_path, d))
    ]

def process_dicom_series(directory):
    """Load DICOM series using SimpleITK"""
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            return None
        
        dicom_files = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
        reader.SetFileNames(dicom_files)
        return reader.Execute()
    except Exception as e:
        print(f"Error loading DICOM from {directory}: {str(e)}")
        return None

def normalize_ct(ct_volume):
    """Normalize CT volume to [-1, 1] range"""
    # Clip to reasonable HU range
    ct_volume = np.clip(ct_volume, -1000, 1000)
    # Normalize to [-1, 1]
    ct_volume = (ct_volume - (-1000)) / (1000 - (-1000)) * 2 - 1
    return ct_volume

def process_files():
    """Process DICOM files and save the processed data.
    Returns the processed data if successful, None otherwise."""
    try:
        processed_data = []
        target_shape = (128, 128, 72)
        base_path = "test_data"
        working_dir = "processed_files"
        os.makedirs(working_dir, exist_ok=True)
        patients = get_patient_folders(base_path)

        if not patients:
            print("No patient folders found in test_data directory")
            return None

        for patient in patients:
            patient_id = os.path.basename(patient)
            print(f"\nProcessing: {patient_id}")
            ct_path = os.path.join(patient, "im_1")
            seg_path = os.path.join(patient, "im_3")
            ct_image = process_dicom_series(ct_path)
            if not ct_image:
                print(f"Skipping {patient_id} - CT load failed")
                continue
            seg_image = process_dicom_series(seg_path)
            if not seg_image:
                print(f"Skipping {patient_id} - Segmentation load failed")
                continue
            try:
                ct_array = sitk.GetArrayFromImage(ct_image)  # (D, H, W)
                seg_array = sitk.GetArrayFromImage(seg_image)
                if len(seg_array.shape) == 4:
                    seg_array = seg_array.squeeze(0)
                ct_tensor = torch.tensor(ct_array).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                seg_tensor = torch.tensor(seg_array).float().unsqueeze(0).unsqueeze(0)
                ct_resized = F.interpolate(ct_tensor, size=target_shape, mode='trilinear').squeeze().numpy()  # (D, H, W)
                seg_resized = F.interpolate(seg_tensor, size=target_shape, mode='nearest').squeeze().numpy()  # (D, H, W)
                processed_data.append({
                    'filename': patient_id,
                    'ct': ct_resized.astype(np.float32),  # (D, H, W)
                    'seg': seg_resized.astype(np.float32),  # (D, H, W)
                    'original_shape': ct_array.shape,
                    'processed_shape': target_shape
                })
            except Exception as e:
                print(f"Error processing {patient_id}: {str(e)}")
                continue

        if not processed_data:
            print("No data was successfully processed")
            return None

        output_path = os.path.join(working_dir, 'complete_dataset.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"Successfully processed and saved {len(processed_data)} samples")
        return processed_data

    except Exception as e:
        print(f"Error in process_files: {str(e)}")
        return None

if __name__ == "__main__":
    processed_data = process_files()
    if processed_data:
        print(f"Processed {len(processed_data)} samples")
    else:
        print("Processing failed")
