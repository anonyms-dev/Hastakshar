#!/usr/bin/env python3
"""
time2vision_transformer.py

Convert time series into image (GAF) and embed with a Vision Transformer.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch

# --------------------------
# Step 1: Load a .txt time series
# --------------------------
def load_timeseries(path, delimiter=None):
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim > 1:   # multiple cols: take first col
        data = data[:, 0]
    return data

# --------------------------
# Step 2: Convert to Gramian Angular Field (vision image)
# --------------------------
def ts_to_gaf(ts, img_size=224, save_path="gaf.png"):
    # normalize
    ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
    gaf = GramianAngularField(image_size=img_size, method='summation')
    X_gaf = gaf.fit_transform(ts.reshape(1, -1))[0]

    # save image
    plt.imsave(save_path, X_gaf, cmap="rainbow")
    print(f"✅ Saved GAF image at {save_path}")
    return save_path

# --------------------------
# Step 3: Pass image through pretrained ViT
# --------------------------
def encode_with_vit(image_path):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    img = Image.open(image_path).convert("RGB")
    
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token embedding (global representation)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    print("Embedding shape:", embedding.shape)
    return embedding

# --------------------------
# Main
# --------------------------
def main():
    # your .txt file with time series
    input_file = "/Users/balaji.raok/Documents/Digital_smell/Dataset/dataset_cofee/AQ_Coffee/AQ_01.txt"
    ts = load_timeseries(input_file)

    # Convert TS → GAF image
    gaf_path = ts_to_gaf(ts, img_size=224, save_path="gaf_Aq.png")

    # Encode with pretrained Vision Transformer
    emb = encode_with_vit(gaf_path)

    print("Final vector representation:", emb.shape)

if __name__ == "__main__":
    main()
