# MultiPass U-Net for Medical Image Segmentation

This repository contains a Jupyter Notebook implementing a **MultiPass U-Net** architecture, designed for enhanced segmentation performance on medical or histopathological images. The multi-pass mechanism allows the model to iteratively refine predictions by processing the input through multiple U-Net passes, improving delineation of fine structures and rare regions.

---

## 📌 Features

- Multi-pass refinement architecture based on the U-Net backbone.
- Supports grayscale and RGB image inputs.
- Designed for histopathology, radiology, and general biomedical segmentation tasks.
- Easy to extend or plug into your existing image processing pipelines.

---

## 🧠 Method Overview

The **MultiPass U-Net** builds upon the classic U-Net model by:
- Running multiple forward passes over the input or intermediate outputs.
- Optionally concatenating features from previous passes for refinement.
- Capturing both coarse and fine features more effectively.

---
