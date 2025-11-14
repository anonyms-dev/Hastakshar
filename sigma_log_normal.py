import numpy as np
import json
import matplotlib.pyplot as plt
from sigma_lognormal.plotting import show_plot
from sigma_lognormal.preprocess import preprocess
from sigma_lognormal.speed_extract import extract_all_lognormals
from sigma_lognormal.beam_search import BeamSearch
from sigma_lognormal.action_plan import ActionPlan


# ----------------------------------------------------------
# 🕒 Utility
# ----------------------------------------------------------
def parse_time_to_ms(tstr: str) -> int:
    """Convert 'H:M:S:microseconds' → milliseconds."""
    h, m, s, us = map(int, tstr.split(":"))
    return ((h * 3600) + (m * 60) + s) * 1000 + us // 1000


# ----------------------------------------------------------
# 🧩 Step 1: Load and parse signature file
# ----------------------------------------------------------
def load_signature(input_path: str) -> np.ndarray:
    """Load and parse signature file -> numpy array [x, y, t(ms)]."""
    t0 = None
    signature = []
    with open(input_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            x, y, t_str = parts[:3]
            x = float(x)
            y = float(y)
            t_ms = parse_time_to_ms(t_str)
            if t0 is None:
                t0 = t_ms
            t_rel = t_ms - t0
            signature.append([x, y, t_rel])
    return np.array(signature, dtype=np.float32)


# ----------------------------------------------------------
# 🧹 Step 2: Preprocess & normalize
# ----------------------------------------------------------
def preprocess_signature(signature: np.ndarray):
    """Preprocess signature for lognormal decomposition."""
    return preprocess(signature)


# ----------------------------------------------------------
# 🎨 Step 3: Plot helpers
# ----------------------------------------------------------
def plot_signature(signal, filename="signature_plot"):
    plt.title("Signature")
    show_plot("signals", [signal], filename)


def plot_speed_profile(signal, filename="speed_profile"):
    plt.title("Signature speed profile")
    plt.xlabel("Time (ms)")
    plt.ylabel("Speed")
    show_plot("speeds", [signal], filename, should_scatter=False)


def plot_recreated_signal(signal, plan, filename="recreated"):
    small_plan = plan.sub_plan(25)
    small_signal = small_plan.signal(signal.time)
    plt.title(f"Recreation from {len(small_plan.strokes)} handstrokes")
    show_plot("signals", [signal, small_signal], filename, legend=["Original", "Recreated"])


def plot_primary_handstrokes(signal, plan, filename="primary_handstrokes", threshold=150):
    def get_modified_stroke(stroke):
        base_signal = stroke.signal(signal.time)
        max_speed_time = stroke.time_of_max_speed()
        base_position_of_max_speed = stroke.position(max_speed_time)
        max_speed_signal_index = signal.find_closest_time(max_speed_time)
        real_position_of_max_speed = signal.position[max_speed_signal_index]
        position_offset = real_position_of_max_speed - base_position_of_max_speed
        return base_signal + position_offset

    stroke_signals = [
        get_modified_stroke(stroke)
        for stroke in plan.strokes
        if stroke.D > threshold
    ]
    # axes[2]
    plt.title("Primary handstrokes in signature")
    show_plot("signals", stroke_signals, filename)


# ----------------------------------------------------------
# 🚀 Step 4: Beam search & stroke reconstruction
# ----------------------------------------------------------
def run_beam_search(signal, top_k=2, snr_threshold=25, max_strokes=30, use_cached=False, cache_file="cached_plan.json"):
    search = BeamSearch(signal, top_k, snr_threshold, max_strokes)
    if not use_cached:
        out_plan, snr_plot = search.search()
        with open(cache_file, "w") as f:
            json.dump(out_plan.to_json(), f)
    else:
        out_plan = ActionPlan.from_json(json.load(open(cache_file)))
    return out_plan


# ----------------------------------------------------------
# 🧠 Step 5: Full pipeline
# ----------------------------------------------------------
def process_signature_pipeline(input_path: str, output_prefix="hindi_signature", plot=True):
    signature = load_signature(input_path)
    signal = preprocess_signature(signature)

    if plot:
        plot_signature(signal, f"{output_prefix}_preprocessed")
        plot_speed_profile(signal, f"{output_prefix}_speed")

    # Extract lognormals
    extract_all_lognormals(signal)

    # Beam Search
    out_plan = run_beam_search(signal, use_cached=False, cache_file=f"{output_prefix}_cached_plan.json")

    if plot:
        plot_recreated_signal(signal, out_plan, f"{output_prefix}_recreated")
        plot_primary_handstrokes(signal, out_plan, f"{output_prefix}_primary_strokes")

    print("✅ Processing complete!")


# ----------------------------------------------------------
# 🧭 Example Usage
# ----------------------------------------------------------
if __name__ == "__main__":
    input_sig = "data_for_plot/sign_u024_O01.txt"
    process_signature_pipeline(input_sig, "hindi_signature")
