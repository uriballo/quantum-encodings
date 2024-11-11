import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from skimage.metrics import structural_similarity as ssim


def run_circuit(circuit, simulator=AerSimulator(), shots=1024):
    """
    Transpile and run a quantum circuit on a specified simulator.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to execute.
        simulator (AerSimulator): Qiskit Aer simulator to use.
        shots (int): Number of shots to execute.

    Returns:
        dict: Count results from the circuit execution.
    """
    transpiled_circuit = transpile(circuit, simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    return result.get_counts(circuit)


def reconstruct_image(dims, n_pixels, counts, shots):
    """
    Reconstruct an image from qubit measurement results.

    Parameters:
        dims (tuple): Desired image dimensions.
        n_pixels (int): Number of pixels in the flattened image.
        counts (dict): Measurement counts from the quantum circuit.
        shots (int): Total number of shots.

    Returns:
        np.ndarray: Reconstructed image as a 2D numpy array.
    """
    if n_pixels != dims[0] * dims[1]:
        raise ValueError("Provided dimensions do not match the number of pixels.")

    pixel_values = np.zeros(n_pixels)
    for bitstring, count in counts.items():
        for i, bit in enumerate(bitstring):
            if bit == '0':
                pixel_values[i] += count
                
    pixel_values /= shots
    
    # Convert pixel values to intensities and normalize to [0, 255]
    intensities = 2 * np.arccos(np.sqrt(pixel_values))
    scaled_intensities = np.interp(intensities, (0, np.pi), (0, 255)).astype(int)
    
    return np.reshape(scaled_intensities, dims)


def scalability(qubit_usage, operations_usage, n_channels=1, n_min=1, n_max=10):
    """
    Calculate the scalability score based on qubit and operation usage.

    Parameters:
        qubit_usage (list): List of qubit counts used per circuit.
        operations_usage (list): List of operation counts used per circuit.
        n_min (int): Minimum exponent for image size.
        n_max (int): Maximum exponent for image size.

    Returns:
        list: Scalability scores for each image size.
    """
    ns = np.arange(n_min, n_max+1)
    return [complexity_score(ops, qbits, n, n_channels) for ops, qbits, n in zip(operations_usage, qubit_usage, ns)]


def complexity_score(depth, n_qubits, n_pixels, n_channels = 1):
    """
    Calculate the complexity score for a given circuit based on depth and qubits used.

    Parameters:
        depth (int): Depth of the circuit (number of operations).
        n_qubits (int): Number of qubits used in the circuit.
        n_pixels (int): The n in 2^n x 2^n image.

    Returns:
        float: Complexity score.
    """
    total = 2 ** n_pixels * 2 ** n_pixels * n_channels

    depth_efficiency = depth / total
    qubit_efficiency = n_qubits / np.log2(total)
    
    return depth_efficiency / (depth_efficiency * qubit_efficiency)


def compute_ssim(image1, image2):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Parameters:
        image1 (np.ndarray): First image array.
        image2 (np.ndarray): Second image array.

    Returns:
        float: SSIM index.
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    ssim_index, _ = ssim(image1, image2, full=True)
    return ssim_index


def retrieval_score(circuit, target_image, dims=(4, 4), shots=1024):
    """
    Calculate the Mean Absolute Error (MAE) between target and reconstructed images.

    Parameters:
        circuit (QuantumCircuit): Quantum circuit representing image encoding.
        target_image (np.ndarray): Target image as a 2D numpy array.
        dims (tuple): Dimensions of the target image.
        shots (int): Number of shots for measurement.

    Returns:
        float: Mean Absolute Error (MAE) between target and reconstructed images.
    """
    target_image = target_image.flatten()
    n_pixels = len(target_image)
    counts = run_circuit(circuit, shots=shots)
    reconstructed_image = reconstruct_image(dims, n_pixels, counts, shots)
    mae = np.abs(target_image - reconstructed_image).mean()
    return mae


def retrieval_efficiency_score(mae, complexity):
    """
    Calculate the retrieval efficiency score as a ratio of MAE to complexity.

    Parameters:
        mae (float): Mean Absolute Error (MAE) of image reconstruction.
        complexity (float): Complexity score of the encoding.

    Returns:
        float: Retrieval efficiency score.
    """
    return mae / complexity
