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

def convert_seg_dicom_to_mha(im3_path, output_dir):
    """Convert DICOM segmentation to MHA format"""
    seg_dcm_files = [f for f in os.listdir(im3_path) if f.lower().endswith(".dcm")]
    if not seg_dcm_files:
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    convert_cmd = [
        "segimage2itkimage",
        "--inputDICOM", os.path.join(im3_path, seg_dcm_files[0]),
        "--outputDirectory", output_dir,
        "-t", "mha",
        "-p", "segmentation"
    ]
    
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Conversion failed for {im3_path}. Error:\n{result.stderr}")
        return None

    mha_files = [f for f in os.listdir(output_dir) if f.startswith("segmentation") and f.endswith(".mha")]
    return os.path.join(output_dir, mha_files[0]) if mha_files else None

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

def preprocess_volume(volume, target_shape):
    """Resample volume to target shape"""
    tensor = torch.tensor(volume).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear')
    return resized.squeeze().numpy()

def process_files():
    processed_data = []
    target_shape = (128, 128, 72)

    # Local paths
    base_path = "ctpel"  # Local dataset directory
    working_dir = "processed_files"  # Local output directory
    os.makedirs(working_dir, exist_ok=True)

    patients = get_patient_folders(base_path)

    for patient in patients:
        patient_id = os.path.basename(patient)
        print(f"\nProcessing: {patient_id}")
        
        ct_path = os.path.join(patient, "im_1")
        seg_path = os.path.join(patient, "im_3")

        # 1. Load CT scan
        ct_image = process_dicom_series(ct_path)
        if not ct_image:
            print(f"Skipping {patient_id} - CT load failed")
            continue

        # 2. Process segmentation
        seg_file = convert_seg_dicom_to_mha(seg_path, working_dir)
        if not seg_file:
            print(f"Skipping {patient_id} - Segmentation conversion failed")
            continue
            
        try:
            # 3. Load and resample segmentation
            seg_image = sitk.ReadImage(seg_file)
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ct_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled_seg = resampler.Execute(seg_image)

            # 4. Convert to numpy arrays
            ct_array = sitk.GetArrayFromImage(ct_image)  # (D, H, W)
            seg_array = sitk.GetArrayFromImage(resampled_seg)

            # 5. Verify dimensions
            if ct_array.shape != seg_array.shape:
                print(f"Shape mismatch after resampling: {ct_array.shape} vs {seg_array.shape}")
                continue

            # 6. Convert to tensors and resize
            ct_tensor = torch.tensor(ct_array).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            seg_tensor = torch.tensor(seg_array).float().unsqueeze(0).unsqueeze(0)

            ct_resized = F.interpolate(ct_tensor, size=target_shape, mode='trilinear').squeeze()
            seg_resized = F.interpolate(seg_tensor, size=target_shape, mode='nearest').squeeze()

            seg_final = (seg_resized > 0.5).float().numpy()

            # 7. Store processed data
            processed_data.append({
                'filename': patient_id,
                'ct': ct_resized.numpy().astype(np.float32),  # (H, W, D)
                'seg': seg_final.astype(np.float32),          # (H, W, D)
                'original_shape': ct_array.shape,
                'processed_shape': target_shape
            })

            # Save individual patient data
            patient_data = {
                'filename': patient_id,
                'ct': ct_resized.numpy().astype(np.float32),
                'seg': seg_final.astype(np.float32),
                'original_shape': ct_array.shape,
                'processed_shape': target_shape
            }
            
            # Save as pickle file
            output_file = os.path.join(working_dir, f"{patient_id}_processed.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(patient_data, f)

        except Exception as e:
            print(f"Error processing {patient_id}: {str(e)}")
            continue

    
    with open(os.path.join(working_dir, 'complete_dataset.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)

    return processed_data

if __name__ == "__main__":
    processed_data = process_files()
    print(f"Processed {len(processed_data)} samples")