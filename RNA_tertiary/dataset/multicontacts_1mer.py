from multiprocessing import Pool
import Bio.PDB
import os 
import numpy as np
import time

def read_pdb(pdb_file):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_file)
    return structure

def generate_pad_cmap_hv(pdb_file,cut_off):
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
                contact_map[i,j] = 1
                contact_map[j,i] = 1
                
    #padded_contact_map = np.pad(contact_map, ((0, 440 - num_residues), (0, 440 - num_residues)), 'constant')
    
    return contact_map

def process_file(file_path):
    pdb_name = os.path.basename(file_path)
    cmap = generate_pad_cmap_hv(file_path, cut_off = 12.0)
    contact_map = np.array(cmap)
    #-----------------------------------------------------------------------------------------
    np.savez_compressed('./1mer_cmaps/contacts/'+pdb_name+'_contacts.npz', b=contact_map)  # Change save location
    #-----------------------------------------------------------------------------------------

if __name__ == '__main__':
    #-----------------------------------------------------------------------------------------
    directory = './pdbs/' # Change input folder
    #-----------------------------------------------------------------------------------------
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

