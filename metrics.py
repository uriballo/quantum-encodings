import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from skimage.metrics import structural_similarity as ssim


def run_circuit(circuit, simulator = AerSimulator(), shots = 1024):
    transpiled_circuit = transpile(circuit, simulator)
    result = simulator.run(transpiled_circuit, shots=shots)
    return result.get_counts(circuit)

def reconstruct_image(dims, n_pixels, counts, shots):
    values = np.zeros(n_pixels)
    for item in counts:
        for i, bit in enumerate(item):
            if bit == '0':
                values[i] += counts[item]
                
    values /= shots
    
    reconstruction = []
    for pixel in values:
        intensity = 2 * np.arccos((pixel) ** (1/2))
        reconstruction.append(intensity)
    
    reconstruction_list = list(np.interp(reconstruction, (0, np.pi), (0, 255)).astype(int))
    return np.reshape(reconstruction_list, dims)    

def complexity_score(operations, n_qubits, n_pixels):
    total = 2**(n_pixels)*2**(n_pixels)
    op_efficiency = operations / total
    qubit_efficiency = n_qubits / total
        
    return op_efficiency + qubit_efficiency

def compute_ssim(image1, image2):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Parameters:
        image1 (np.ndarray): First image as a numpy array.
        image2 (np.ndarray): Second image as a numpy array.

    Returns:
        float: SSIM index between the two images.
    """
    # Ensure the images are in the same shape
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Compute SSIM
    ssim_index, _ = ssim(image1, image2, full=True)
    return ssim_index

def retrieval_score(circuit, image, dims = (4,4), shots = 1024):
    counts = run_circuit(circuit, shots=shots)
        
    # func2 Reconstruct image
    image = image.flatten
    n_pixels = len(image)
    values = np.zeros(n_pixels)
    for item in counts:
        for i, bit in enumerate(item):
            if bit=='0':
                values[i]+=counts[item]

    values = values/shots

    reconstruct = []
    for pixel in values:
        color = 2*np.arccos((pixel)**(1/2)) # "shots" corresponds to the total counts value.
        reconstruct.append(color)
        
    reconstruct = list(np.interp(reconstruct, (0,np.pi), (0,255)).astype(int))
    reconstruct = reconstruct_image(dims, n_pixels, counts, shots)

    mae = np.abs(image - reconstruct).mean(axis=None)
    return mae 

def retrieval_efficiency_score(mae, complexity):
    return mae / complexity
