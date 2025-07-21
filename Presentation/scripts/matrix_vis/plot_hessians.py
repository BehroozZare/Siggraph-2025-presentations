import os
import glob
import matplotlib.pyplot as plt
from scipy.io import mmread
import numpy as np
# Directory containing the hessian .mtx files
hessian_dir = os.path.join(os.path.dirname(__file__), 'hessians', 'hessian_checkpoints')
# hessian_dir = os.path.join(os.path.dirname(__file__), 'hessians')
output_dir = os.path.join(os.path.dirname(__file__), 'results')

os.makedirs(output_dir, exist_ok=True)

# Find all .mtx files
mtx_files = glob.glob(os.path.join(hessian_dir, '*.mtx'))
print(mtx_files)
#Sort based on the number in the file name
mtx_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
for mtx_file in mtx_files[0:200]:
    print(f'Reading {mtx_file}...')
    matrix = mmread(mtx_file)
    sparse_matrix = matrix.tocsc()
    #Print the sparsity ratio (number of zeros / total number of elements)
    sparsity_ratio = (1 - sparse_matrix.count_nonzero() / (matrix.shape[0] * matrix.shape[1])) * 100
    print(f'Sparsity ratio: {sparsity_ratio}')
    plt.figure(figsize=(8, 8))
    plt.spy(matrix, markersize=0.5)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0)
    out_path = os.path.join(output_dir, os.path.basename(mtx_file).replace('.mtx', '_spy.png'))
    plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Saved spy plot to {out_path}')
