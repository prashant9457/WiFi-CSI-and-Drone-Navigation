import scipy.io as scio
import numpy as np
import os

# Adjust this filename to match an exact file you unzipped in your directory
sample_file = "code/data/1-1-1-1-1.mat" 

if os.path.exists(sample_file):
    matrix_dict = scio.loadmat(sample_file)
    print("Available keys in this MAT file:", [k for k in matrix_dict.keys() if not k.startswith('__')])
    
    # Widar 3.0 BVP files store the matrix under the key 'f_bvp' or 'bvp'
    bvp_key = 'f_bvp' if 'f_bvp' in matrix_dict else 'bvp'
    matrix_data = matrix_dict[bvp_key]
    
    print(f"BVP Matrix Shape: {matrix_data.shape}")
else:
    print(f"File not found at {sample_file}. Please check your path and unzipped filenames!")