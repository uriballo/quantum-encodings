from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from metrics import complexity_score


def plot_bw_image(path, title=""):
    image = Image.open(path)
    image = np.array(image)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(title)
    plt.show()
    return image

def plot_rgb_image(path, title=""):
    image = Image.open(path)
    image = np.array(image)
    
    plt.imshow(image, vmin=0, vmax=255)

    plt.axis('off')
    plt.title(title)
    plt.show()
    
    return image

def plot_complexity_scaling(operations_scaling, qubit_scaling, two_n_min=1, two_n_max=10):
    # Define the range of side lengths
    # 2^n x 2^n image 
    # metrics are in terms of n
    n_values = np.arange(two_n_min, two_n_max) # images from 2x2 to 1024x1024 
    
    # Calculate the metric values and the complexity score
    operations_values = np.array([operations_scaling(n) for n in n_values])
    qubit_values = np.array([qubit_scaling(n) for n in n_values])
    C_values = np.array([complexity_score(ops, qubits, pixels) for ops, qubits, pixels 
                         in zip(operations_values, qubit_values, n_values)])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, C_values, label='Complexity Score', linewidth=2)
    plt.title('Scaling of Complexity Score with Image Size')
    plt.xlabel(r'Image Size $(2^n \times 2^n)$')
    plt.ylabel('Complexity Score')
    #plt.xscale('linear')
    #plt.yscale('log')  
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_complexity_scaling_comparison(operations_scaling_1, qubit_scaling_1, encoding_name1,
                                       operations_scaling_2, qubit_scaling_2, encoding_name2,
                                       two_n_min=1, two_n_max=10, logscale=False):
    # Define the range of side lengths (2^n x 2^n image sizes)
    n_values = np.arange(two_n_min, two_n_max)
    
    # Calculate the metric values and complexity scores for the first scaling
    operations_values_1 = np.array([operations_scaling_1(n) for n in n_values])
    qubit_values_1 = np.array([qubit_scaling_1(n) for n in n_values])
    C_values_1 = np.array([complexity_score(ops, qubits, n) 
                           for ops, qubits, n in zip(operations_values_1, qubit_values_1, n_values)])
    
    # Calculate the metric values and complexity scores for the second scaling
    operations_values_2 = np.array([operations_scaling_2(n) for n in n_values])
    qubit_values_2 = np.array([qubit_scaling_2(n) for n in n_values])
    C_values_2 = np.array([complexity_score(ops, qubits, n) 
                           for ops, qubits, n in zip(operations_values_2, qubit_values_2, n_values)])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, C_values_1, label=f'{encoding_name1} Complexity Score', linewidth=2)
    plt.plot(n_values, C_values_2, label=f'{encoding_name2} Complexity Score', linewidth=2, linestyle='--')
    plt.title('Comparison of Complexity Scaling')
    plt.xlabel(r'Image Size $(2^n \times 2^n)$')
    plt.ylabel('Complexity Score')    
    if logscale:
        plt.xscale('linear')
        plt.yscale('log')  
    plt.grid(True)
    plt.legend()
    plt.show()
    

def get_angle_representation(image, min_val=0, max_val=255):
    """
    Converts the pixel intensities of an image to angles between 0 and pi.
    
    Parameters:
        image (np.ndarray): Input image as a numpy array.
        min_val (float): Minimum possible pixel intensity value in the image.
        max_val (float): Maximum possible pixel intensity value in the image.
        
    Returns:
        np.ndarray: 1D array of angles representing the image.
    """

    flat_image = image.flatten()
    angles = (flat_image - min_val) * (np.pi / (max_val - min_val))
    
    return angles