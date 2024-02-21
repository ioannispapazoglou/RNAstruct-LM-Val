from multiprocessing import Pool
from tqdm import tqdm  # Import tqdm
import Bio.PDB
import os
import numpy as np
import time

def read_pdb(pdb_file):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_file)
    return structure

def generate_cmap_hv(pdb_file, cut_off):
    pdb = read_pdb(pdb_file)
    structure = pdb[0]
    residues = [res for res in structure.get_residues() if res.id[0] == " "]
    num_residues = len(residues)
    contact_map = np.zeros((num_residues, num_residues))

    for i in range(num_residues):
        res1 = residues[i]
        for j in range(i+1, num_residues):
            res2 = residues[j]
            min_dist = float('inf')
            for atom1 in res1.get_atoms():
                if atom1.element != 'H':
                    for atom2 in res2.get_atoms():
                        if atom2.element != 'H':
                            dist = atom1 - atom2
                            if dist < min_dist:
                                min_dist = dist
            if min_dist <= cut_off:
                contact_map[i, j] = 1
                contact_map[j, i] = 1

    return contact_map

def process_file(file_path):
    pdb_name = os.path.basename(file_path)
    cmap = generate_cmap_hv(file_path, cut_off=9.5) # Set cutoff
    contact_map = np.array(cmap)
    np.savez_compressed('./'+pdb_name+'_contacts.npz', b=contact_map)  # Change save location

if __name__ == '__main__':

    directory = './'  # Set directory containing the pdb files


    file_list = sorted(os.listdir(directory))

    # Wrap the entire file_list in tqdm to monitor overall progress
    with tqdm(total=len(file_list), desc="Processing files") as pbar:
        def update_pbar(*_):
            pbar.update()
        
        # Create a process pool with the specified number of processes
        with Pool(processes=28, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
            for file_name in file_list:
                file_path = os.path.join(directory, file_name)
                p.apply_async(process_file, args=(file_path,), callback=update_pbar)

            p.close()
            p.join()

    print("Done. Bye!")
