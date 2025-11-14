# CVPR plot 
import numpy as np
import json
import scipy
import torch
import matplotlib.pyplot as plt
from sigma_lognormal.plotting import *

from sigma_lognormal.preprocess import preprocess

# with open("data/signatures.txt") as f:
#  signatures = json.load(f)
#  np_signatures = [np.array(signature) for signature in signatures]

def parse_time_to_ms(tstr: str) -> int:
    """
    Convert 'H:M:S:microseconds' -> milliseconds.
    Example: '4:28:22:278630'
    """
    h, m, s, us = tstr.split(":")
    h = int(h); m = int(m); s = int(s); us = int(us)
    return ((h * 3600) + (m * 60) + s) * 1000 + us // 1000


input_sig = "data_for_plot/sign_u024_O01.txt"
# input_sig = "u0001_g_0100v00.txt"
# input_sig = "u0001_s_0100f02.txt"
# input_sig = "u0001_s_0100f18.txt"
t0 = None  # reference time of first sample

signature = []
with open(input_sig) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # split on comma: x, y, time_str, pen/pressure
        parts = line.split(',')
        if len(parts) < 3:
            continue

        x_str, y_str, t_str = parts[:3]

        x = float(x_str)
        y = float(y_str)               # same as your old logic: flip Y
        t_ms = parse_time_to_ms(t_str)  # absolute ms

        if t0 is None:
            t0 = t_ms                   # first point as time zero
        t_rel = t_ms - t0               # relative time

        signature.append(np.array([x, y, t_rel], dtype=np.float32))


signature = np.array(signature)


signals = [preprocess(signature)]

signal = signals[0]
plt.title("Signature")
filename1='hindi_signature_preprocessed'
show_plot("signals",[signal],filename1)