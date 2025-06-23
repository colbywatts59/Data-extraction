import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


output_folder = "augmented_peaks"

def add_noise(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise


def time_warp(signal, warp_factor=1.1):
    length = len(signal)
    new_length = int(length * warp_factor)
    f = interp1d(np.linspace(0, 1, length), signal, kind='linear')
    new_signal = f(np.linspace(0, 1, new_length))
    f2 = interp1d(np.linspace(0, 1, new_length), new_signal, kind='linear')
    return f2(np.linspace(0, 1, length))

for file in os.listdir("individual_peaks"):
    if file.endswith(".csv"):
        file_path = os.path.join("individual_peaks", file)
        df = pd.read_csv(file_path)
        button_name = file.split("_")[0]
        peaks = df['Peak Data'].apply(lambda x: np.array(x.split(",")).astype(float))

        augmented_peaks = []
        augmented_labels = []

        for peak in peaks:
            augmented_peaks.append(peak)  
            augmented_labels.append("original")

            augmented_peaks.append(add_noise(peak))
            augmented_labels.append("noise")

            augmented_peaks.append(time_warp(peak))
            augmented_labels.append("time_warp")

        augmented_strings = [",".join(map(str, p)) for p in augmented_peaks]

        augmented_df = pd.DataFrame({
            'Peak Data': augmented_strings,
            'Augmentation': augmented_labels
        })


        output_path = os.path.join(output_folder, file)
        print(output_path)
        augmented_df.to_csv(output_path, index=False)


