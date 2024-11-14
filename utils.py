from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from metrics import complexity_score, scalability

def line_plot(title, metrics, labels, xlabel, ylabel, save_path = "", filename = "plot",
              save_plot = False, show_plot = True):
    """
    Plots multiple metrics over epochs with different colors and markers.

    Args:
        title (str): The main title of the plot.
        metrics (list of lists): A list containing lists of metric values per epoch.
        labels (list of str): A list of labels corresponding to each metric.
        save_path (str): Directory path to save the plot.
        filename (str): Filename for the saved plot.
    """
    epochs = range(1, len(metrics[0]) + 1)

    plt.figure(figsize=(8, 8))
    
    # Define colors and markers for plotting
    colors = plt.cm.bwr(np.linspace(0, 1, len(metrics)))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'H', '<', '>']

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.plot(epochs, metric, color=colors[i], marker=markers[i % len(markers)], 
                 label=label, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(epochs, minor=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_path + filename + '.png', bbox_inches='tight')
        
    if show_plot:
        plt.show()

def collect_metrics(encoding_func, n_min=1, n_max=8, n_channels=1):
    qubits_usage, depth = [], []

    image_sizes = [2 ** n for n in range(n_min, n_max + 1)]
    
    # Generate random images with varying sizes and channels
    images = [np.random.rand(size, size) if n_channels == 1 else np.random.rand(size, size, n_channels) for size in image_sizes]
    
    # Collect metrics by applying encoding function on each image
    for img in images:
        circuit = encoding_func(img)
        qubits_usage.append(circuit.num_qubits)
        depth.append(circuit.depth())
    
    return qubits_usage, depth

def plot_scalability(encoding_func, n_channels=1):
    qubits_usage, depth = collect_metrics(encoding_func, n_channels=n_channels)
    scaling = scalability(qubits_usage, depth, n_channels=n_channels)
    
    plot_scaling(scaling)

    return scaling

def plot_image(path, title="", rgb = False, vmin=0, vmax=255):
    image = np.array(Image.open(path))
    if rgb:
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)

    plt.axis('off')
    plt.title(title)
    plt.show()
    return image

def plot_scaling(complexity_vals, two_n_min=1, two_n_max=8):
    n_values = np.arange(two_n_min, two_n_max + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, complexity_vals, label='Complexity Score', linewidth=2)
    plt.title('Scaling of Complexity Score with Image Size')
    plt.xlabel(r'Image Size $(2^n \times 2^n)$')
    plt.ylabel('Complexity Score')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_complexity_scaling(depth_scaling, qubit_scaling, two_n_min=1, two_n_max=8, logscale=False):
    n_values = np.arange(two_n_min, two_n_max + 1)
    print(n_values)
    # Compute complexity score based on scaling
    operations_values = np.array([depth_scaling(n) for n in n_values])
    qubit_values = np.array([qubit_scaling(n) for n in n_values])
    complexity_scores = np.array([complexity_score(ops, qubits, pixels) for ops, qubits, pixels in zip(operations_values, qubit_values, n_values)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, complexity_scores, label='Complexity Score', linewidth=2)
    plt.title('Scaling of Complexity Score with Image Size')
    plt.xlabel(r'Image Size $(2^n \times 2^n)$')
    plt.ylabel('Complexity Score')
    
    if logscale:
        plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_complexity_scaling_comparison(scaling_1, encoding_name1, scaling_2, encoding_name2, two_n_min=1, two_n_max=8, logscale=False):
    n_values = np.arange(two_n_min, two_n_max+1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, scaling_1, label=f'{encoding_name1} Complexity Score', linewidth=2)
    plt.plot(n_values, scaling_2, label=f'{encoding_name2} Complexity Score', linewidth=2, linestyle='--')
    plt.title('Comparison of Complexity Scaling')
    plt.xlabel(r'Image Size $(2^n \times 2^n)$')
    plt.ylabel('Complexity Score')
    
    if logscale:
        plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()


def get_angle_representation(image, min_val=0, max_val=255):
    """
    Convert pixel intensities to angles between 0 and π.
    """
    flat_image = image.flatten()
    angles = (flat_image - min_val) * (np.pi / (max_val - min_val))
    
    return angles
