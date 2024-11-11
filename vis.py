import os
import numpy as np
import wfdb
from matplotlib import pyplot as plt

def is_target(symbol: str) -> bool:
    return symbol in ["N", "A", "V", "L", "R"]
# Visualization of a random sample from the processed sliding window data with annotation markers
# Visualization of a random sample from the processed sliding window data
def visualize_random_window(label: str):
    # Load the saved .npy file for the specified label
    file_path = f'./assets/{label}.npy'
    if not os.path.exists(file_path):
        print(f"No data found for label {label}")
        return

    # Load data and pick a random window
    data = np.load(file_path)
    random_index = np.random.randint(data.shape[0])
    random_window = data[random_index]

    # Plot the selected window
    plt.figure(figsize=(10, 4))
    plt.plot(random_window)
    plt.title(f"Random Sliding Window Sample for Label '{label}'")
    plt.xlabel("Sample")
    plt.ylabel("Normalized Amplitude")
    plt.show()


# Example: Visualize a random window from label "N"
visualize_random_window("N")