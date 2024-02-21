import os
import numpy as np
from tqdm import tqdm

input_directory = './'  # Replace with the contacts directory
output_directory = './pdb_nodiag/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

npz_files = [f for f in os.listdir(input_directory) if f.endswith('.npz')]

for npz_file in tqdm(npz_files):

    with np.load(os.path.join(input_directory, npz_file)) as data:
        original_matrix = data['b'] # Adjust key

    modified_matrix = original_matrix.copy()
    for i in range(6): # Set nearby contacts range (+-6)
        np.fill_diagonal(modified_matrix[i:], 0)
        np.fill_diagonal(modified_matrix[:, i:], 0)

    output_file = os.path.join(output_directory, npz_file)
    np.savez(output_file, b = modified_matrix)

print("Done. Bye!")
