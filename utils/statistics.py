import os
import numpy as np
import matplotlib.pyplot as plt

class Statistic:
    def __init__(self, min, max, bins):
        self.min = min
        self.max = max
        self.bins = bins
        self.data = []  # 存储所有输入数据

    def update(self, x):
        flattened = np.array(x).flatten()
        self.data.extend(flattened)

    def compute_histogram(self):
        data_array = np.array(self.data)
        min_val = self.min if self.min is not None else data_array.min()
        max_val = self.max if self.max is not None else data_array.max()
        hist, bin_edges = np.histogram(
            data_array,
            bins=self.bins,
            range=(min_val, max_val)
        )
        return hist, bin_edges

    def plot_and_save(self, data_name, save_path):
        filename = os.path.join(save_path, data_name) + '.png'
        hist, bin_edges = self.compute_histogram()
        plt.figure(figsize=(10, 6))
        plt.bar(
            bin_edges[:-1],        
            hist / len(self.data),
            width=np.diff(bin_edges),
            align='edge',           
            edgecolor='black'       
        )
        plt.title(f"{data_name} Distribution (n={len(self.data)})")
        plt.xlabel(f"{data_name}")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.5)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()