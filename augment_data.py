import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


output_folder = "augmented_peaks"

# Seed for reproducibility
np.random.seed(42)


def add_noise(signal, snr_scale=0.1):
    s = np.std(signal) + 1e-6
    noise = np.random.normal(0, snr_scale * s, signal.shape)
    return signal + noise


def time_warp(signal, warp_min=0.9, warp_max=1.1):
    warp_factor = np.random.uniform(warp_min, warp_max)
    length = len(signal)
    new_length = max(1, int(length * warp_factor))
    f = interp1d(np.linspace(0, 1, length), signal, kind='linear')
    warped = f(np.linspace(0, 1, new_length))
    f2 = interp1d(np.linspace(0, 1, new_length), warped, kind='linear')
    return f2(np.linspace(0, 1, length))


def time_shift(signal, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return signal
    if shift > 0:
        return np.r_[np.full(shift, signal[0]), signal[:-shift]]
    else:
        shift = -shift
        return np.r_[signal[shift:], np.full(shift, signal[-1])]


# Scan counts per class to define target balancing
class_counts = {}
for file in os.listdir("individual_peaks"):
    if file.endswith(".csv"):
        button_name = file.split("_")[0]
        df = pd.read_csv(os.path.join("individual_peaks", file))
        num = len(df)
        class_counts[button_name] = class_counts.get(button_name, 0) + num

# Target to the max observed per-class count
if len(class_counts) > 0:
    target_per_class = max(class_counts.values())
else:
    target_per_class = 0

for file in os.listdir("individual_peaks"):
    if file.endswith(".csv"):
        file_path = os.path.join("individual_peaks", file)
        df = pd.read_csv(file_path)
        button_name = file.split("_")[0]
        peaks = df['Peak Data'].apply(lambda x: np.array(x.split(",")).astype(float))

        augmented_peaks = []
        augmented_labels = []

        # Determine oversampling factor per file based on class count
        num_orig = len(peaks)
        if num_orig == 0:
            repeats = 0
        else:
            repeats = max(1, int(np.ceil(target_per_class / num_orig)))

        for peak in peaks:
            for _ in range(repeats):
                # Original
                augmented_peaks.append(peak)
                augmented_labels.append("original")
                # Noise
                augmented_peaks.append(add_noise(peak, snr_scale=0.1))
                augmented_labels.append("noise")
                # Time warp (random compress/expand)
                augmented_peaks.append(time_warp(peak, warp_min=0.9, warp_max=1.1))
                augmented_labels.append("time_warp")
                # Time shift (small)
                augmented_peaks.append(time_shift(peak, max_shift=10))
                augmented_labels.append("time_shift")

        augmented_strings = [",".join(map(str, p)) for p in augmented_peaks]

        augmented_df = pd.DataFrame({
            'Peak Data': augmented_strings,
            'Augmentation': augmented_labels
        })

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file)
        print(output_path)
        augmented_df.to_csv(output_path, index=False)


