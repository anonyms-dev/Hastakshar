#!/usr/bin/env python3
"""
Convert multi-column time series (.txt) into Vision Transformer embeddings.

Steps:
 1. Load each .txt file (multi-column time series).
 2. Convert it into a Gramian Angular Field (GAF) image.
 3. Pass image through pretrained ViT.
 4. Save embeddings (one per file) into embeddings.npy and embeddings.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from pyts.image import GramianAngularField
import torch
import timm
from torchvision import transforms
# from torchvision.transforms import functional as F
from PIL import Image
# import pandas as pd

# -------------------------------
# Config
# -------------------------------
DATA_FOLDER = "/Users/balaji.raok/Documents/Digital_smell/Dataset/dataset_cofee/LQ_Coffee"   # path to your folder with .txt files
OUT_DIR = "vision_embeddings_LQ"
MODEL_NAME = "vit_base_patch16_224"  # from timm

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# Load ViT model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)  # no classifier head
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
# transform = transforms.Compose([
#     transforms.Resize((256,192)),
#     transforms.ToTensor(),  # PIL → torch.FloatTensor in [0,1]
#     transforms.Normalize((0.5,), (0.5,))
# ])

# from torchvision.transforms.functional import pil_to_tensor

# def transform(img):
#     # Returns a float tensor in [0,1], CHW
#     return pil_to_tensor(img).float() / 255.0


# -------------------------------
# Helper functions
# -------------------------------
def load_time_series(path):
    """Load multi-column .txt file as numpy array (T, D)."""
    data = np.loadtxt(path, delimiter=None)  # auto detect whitespace/comma
    if data.ndim == 1:
        data = data[:, None]
    return data

def time_series_to_gaf(ts, image_size=64):
    """Convert multi-column time series (T, D) → average over columns → GAF image."""
    # Average across columns (combine features into one signal)
    signal = ts.mean(axis=1)

    # Normalize to [0,1]
    signal_min, signal_max = np.min(signal), np.max(signal)
    if signal_max - signal_min > 1e-8:
        signal = (signal - signal_min) / (signal_max - signal_min)
    else:
        signal = np.zeros_like(signal)

    # GAF encoding
    gaf = GramianAngularField(method="summation", image_size=image_size)
    X_gaf = gaf.fit_transform(signal.reshape(1, -1))[0]
    return X_gaf

def gaf_to_embedding(gaf_img):
    """Convert GAF numpy image → ViT embedding vector."""
    img = Image.fromarray((255 * gaf_img).astype(np.uint8))
    img = img.convert("RGB")  # 3 channels
    # tensor = F.to_tensor(img)
    # tensor = F.resize(tensor, [224, 224])
    # tensor = tensor.unsqueeze(0)
    # print(np.shape(img))
    # print(img)
    # img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    # img = img.view(img.size[1], img.size[0], len(img.getbands()))  # HWC
    # img = img.permute(2, 0, 1)  # CHW
    # print(img)

    

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x).cpu().numpy().flatten()
    return emb

# -------------------------------
# Main loop
# -------------------------------
def main():
    txt_files = sorted(glob(os.path.join(DATA_FOLDER, "*.txt")))
    if len(txt_files) == 0:
        raise SystemExit(f"No .txt files found in {DATA_FOLDER}")

    embeddings = []
    file_names = []

    for f in txt_files:
        print(f)
        temp = os.path.basename(f)
        temp1 = os.path.splitext(temp)
        print(temp1[0])
        # exit()
        print(f"Processing {f} ...")
        
        ts = load_time_series(f)
        print(ts)
        # exit()
        gaf_img = time_series_to_gaf(ts, image_size=224)

        # Save image for inspection
        plt.imsave(os.path.join(OUT_DIR, temp1[0] + ".jpg"), gaf_img, cmap="rainbow")

        # Extract ViT embedding
        emb = gaf_to_embedding(gaf_img)
        # print(emb)

        # embeddings.append(emb)
        file_names.append(os.path.basename(f))

        embeddings = np.array(embeddings)
        np.save(os.path.join(OUT_DIR, temp1[0] + ".npy"), embeddings)

    # Save as CSV
    # df = pd.DataFrame(embeddings, index=file_names)
    # df.to_csv(os.path.join(OUT_DIR, "embeddings.csv"))
    # print(f"Saved embeddings: {embeddings.shape} → {OUT_DIR}/embeddings.npy & .csv")

if __name__ == "__main__":
    main()
