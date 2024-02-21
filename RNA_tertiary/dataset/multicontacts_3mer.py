from multiprocessing import Pool
import Bio.PDB
import os 
import numpy as np
import time

def generate_3mer_cmap_hv(pdb_file, cut_off):
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure("pdb", pdb_file)
    residues = [res for res in structure.get_residues() if res.id[0] == " "]
    
    # Define RNA residue names and their one-letter codes
    rna_dict = {"A": "A", "G": "G", "C": "C", "U": "U"}
    # Create a string of the sequence from the residues
    sequence = ''.join([rna_dict.get(res.get_resname(), "X") for res in residues])
    #print(sequence)
    # Split the sequence into 3-mers
    #sequence_3mers = [sequence[i:i+3] for i in range(0, len(sequence), 1)] #Has 2-1mers too
    sequence_3mers = [sequence[i:i+3] for i in range(0, len(sequence)-2, 1)] #Do step 3 for no coverance..
    #print(sequence_3mers)
    num_residues = len(sequence_3mers)
    contact_map = np.zeros((num_residues, num_residues))

    for i in range(num_residues):
        res1 = sequence_3mers[i]
        for j in range(i+1, num_residues):
            res2 = sequence_3mers[j]
            min_dist = float('inf')
            for atom1 in residues[i].get_atoms():
                if atom1.element != 'H':
                    for atom2 in residues[j].get_atoms():
                        if atom2.element != 'H':
                            dist = atom1 - atom2
                            if dist < min_dist:
                                min_dist = dist

            if min_dist <= cut_off:
                contact_map[i, j] = 1
                contact_map[j, i] = 1
    
#    padded_contact_map = np.pad(contact_map, ((0, 0), (0, 0)), mode='constant')
#    padded_contact_map = np.pad(contact_map, ((0, pad_to - num_residues), (0, pad_to - num_residues)), 'constant', constant_values=((0,0),(0,0)))
    
    return contact_map

def process_file(file_path):
    pdb_name = os.path.basename(file_path)
    cmap = generate_3mer_cmap_hv(file_path, cut_off = 9.5)
    contact_map = np.array(cmap)
    #--------------------------------------------------------------------------------------------------
    np.savez_compressed('./3mer_cmaps/contacts/'+pdb_name+'_contacts.npz', b=contact_map)
    #--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #--------------------------------------------------------------------------------------------------
    directory = './pdbs/' # Change input directory 
    #--------------------------------------------------------------------------------------------------
    file_list = sorted(os.listdir(directory))

    # split the list of files into smaller chunks
    num_processes = 20
    chunk_size = len(file_list) // num_processes
    chunks = [file_list[i:i+chunk_size] for i in range(0, len(file_list), chunk_size)]
    
    print('Working..')
    start = time.time()
    
    # create a process pool with the specified number of processes
    with Pool(num_processes) as p:
        for chunk in chunks:
            for file_name in chunk:
                file_path = os.path.join(directory, file_name)
                p.apply_async(process_file, args=(file_path,))

        p.close()
        p.join()

    end = time.time()
    print("Done. Bye!")

