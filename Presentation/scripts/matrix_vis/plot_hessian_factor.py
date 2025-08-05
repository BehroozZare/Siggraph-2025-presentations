import os
# Fix OpenMP library conflict error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import matplotlib.pyplot as plt
from scipy.io import mmread
import numpy as np
from scipy.sparse import csc_matrix, tril, eye
from sksparse.cholmod import cholesky, cholesky_AAt, CholmodError
from scipy.sparse.csgraph import reverse_cuthill_mckee
try:
    import metis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    print("METIS not available, using other ordering methods")

def validate_and_preprocess_matrix(A):
    """
    Validate and preprocess matrix for CHOLMOD compatibility
    """
    print(f"Matrix validation - Shape: {A.shape}, Format: {A.format}, dtype: {A.dtype}")
    
    # Ensure square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for Cholesky decomposition")
    
    # Convert to CSC format (required by CHOLMOD)
    if A.format != 'csc':
        print("Converting to CSC format...")
        A = A.tocsc()
    
    # Ensure float64 dtype
    if A.dtype != np.float64:
        print("Converting to float64...")
        A = A.astype(np.float64)
    
    # Properly construct symmetric matrix from lower triangular part
    print("Constructing symmetric matrix...")
    L = tril(A)  # Extract lower triangular part
    U = tril(A, k=-1)  # Extract strictly lower triangular (without diagonal)
    A_symmetric = L + U.T  # Lower triangular + transpose of strictly lower triangular
    A_symmetric.eliminate_zeros()
    
    print(f"Symmetric matrix constructed: {A_symmetric.nnz:,} non-zeros")
    
    # Verify symmetry
    max_diff = np.abs((A_symmetric - A_symmetric.T).data).max() if (A_symmetric - A_symmetric.T).nnz > 0 else 0.0
    print(f"Symmetry verification - max difference: {max_diff:.2e}")
    
    # Use only lower triangular part for CHOLMOD
    print("Extracting lower triangular part for CHOLMOD...")
    A_final = tril(A_symmetric)
    A_final.eliminate_zeros()
    
    # Check for positive definiteness by examining diagonal
    diag_vals = A_final.diagonal()
    min_diag = np.min(diag_vals)
    max_diag = np.max(diag_vals)
    print(f"Diagonal range: [{min_diag:.2e}, {max_diag:.2e}]")
    
    # Check for zeros on diagonal
    zero_diag_count = np.sum(diag_vals == 0)
    if zero_diag_count > 0:
        print(f"Warning: {zero_diag_count} zeros found on diagonal")
    
    if min_diag <= 0:
        print(f"Matrix may not be positive definite (min diagonal: {min_diag})")
        print("Adding regularization...")
        reg_strength = max(abs(min_diag) + 1e-6, 1e-8)
        reg_matrix = reg_strength * eye(A_final.shape[0], format='csc')
        A_final = A_final + reg_matrix
        print(f"Added regularization: {reg_strength:.2e}")
        
        # Update diagonal info
        new_min_diag = np.min(A_final.diagonal())
        print(f"New min diagonal: {new_min_diag:.2e}")
    
    print(f"Final matrix: {A_final.nnz:,} non-zeros, format: {A_final.format}")
    return A_final

def perform_cholesky_factorization(A, ordering_method="natural"):
    """
    Perform sparse Cholesky factorization using CHOLMOD with robust error handling
    """
    try:
        # Perform factorization
        factor = cholesky(A, ordering_method="amd")
        L = factor.L()
        
        print(f"Factorization successful!")
        return L, factor
        
    except CholmodError as e:
        print(f"CHOLMOD-specific error with {ordering_method} ordering:")
        print(f"  Error message: {e}")
        print(f"  This usually indicates:")
        print(f"    - Matrix is not positive definite")
        print(f"    - Matrix has structural issues")
        print(f"    - Ordering method failed")
        return None, None
        
    except ValueError as e:
        print(f"Value error with {ordering_method} ordering:")
        print(f"  Error message: {e}")
        print(f"  This usually indicates matrix format or data type issues")
        return None, None
        
    except MemoryError as e:
        print(f"Memory error with {ordering_method} ordering:")
        print(f"  Error message: {e}")
        print(f"  Matrix may be too large for available memory")
        return None, None
        
    except Exception as e:
        print(f"Unexpected error with {ordering_method} ordering:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {e}")
        return None, None

# Directory containing the hessian .mtx files
hessian_dir = os.path.join(os.path.dirname(__file__), 'hessians', 'hessian_checkpoints')
# hessian_dir = os.path.join(os.path.dirname(__file__), 'hessians')
output_dir = os.path.join(os.path.dirname(__file__), 'results')

os.makedirs(output_dir, exist_ok=True)

# Find all .mtx files
mtx_files = glob.glob(os.path.join(hessian_dir, '*.mtx'))
#Sort based on the number in the file name
mtx_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

print(f"Found {len(mtx_files)} matrix files")

for mtx_file in mtx_files[0:1]:
    print(f'\n{"="*60}')
    print(f'Processing: {os.path.basename(mtx_file)}')
    print(f'{"="*60}')
    
    try:
        matrix = mmread(mtx_file)
        sparse_matrix = matrix.tocsc()
        
        #Print basic matrix info
        sparsity_ratio = (1 - sparse_matrix.count_nonzero() / (sparse_matrix.shape[0] * sparse_matrix.shape[1])) * 100
        print(f'Original matrix info:')
        print(f'  Size: {sparse_matrix.shape[0]} x {sparse_matrix.shape[1]}')
        print(f'  Non-zeros: {sparse_matrix.count_nonzero():,}')
        print(f'  Sparsity: {sparsity_ratio:.2f}%')
        print(f'  Format: {sparse_matrix.format}')
        print(f'  Data type: {sparse_matrix.dtype}')
        
        # Original matrix spy plot
        plt.figure(figsize=(8, 8))
        plt.spy(matrix, markersize=0.5)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.title('Original Matrix')
        plt.tight_layout(pad=0)
        out_path = os.path.join(output_dir, os.path.basename(mtx_file).replace('.mtx', '_spy.png'))
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Saved original matrix spy plot to {out_path}')
        
        # Cholesky factorization with natural ordering (no reordering)
        print(f'\n{"-"*40}')
        print("NATURAL ORDERING")
        print(f'{"-"*40}')
        L_natural, factor_natural = perform_cholesky_factorization(sparse_matrix, ordering_method="natural")
        
        if L_natural is not None:
            plt.figure(figsize=(8, 8))
            plt.spy(L_natural, markersize=0.5)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.title('Cholesky Factor L - Natural Ordering')
            plt.tight_layout(pad=0)
            out_path = os.path.join(output_dir, os.path.basename(mtx_file).replace('.mtx', '_cholesky_natural_ordering.png'))
            plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f'Saved Cholesky factor (natural ordering) to {out_path}')
            
            factor_sparsity = (1 - L_natural.count_nonzero() / (L_natural.shape[0] * L_natural.shape[1])) * 100
            print(f'Results:')
            print(f'  Factor non-zeros: {L_natural.count_nonzero():,}')
            print(f'  Factor sparsity: {factor_sparsity:.2f}%')
        
        # Cholesky factorization with METIS ordering
        ordering_method = "metis" if METIS_AVAILABLE else "amd"
        print(f'\n{"-"*40}')
        print(f"{ordering_method.upper()} ORDERING")
        print(f'{"-"*40}')
        L_ordered, factor_ordered = perform_cholesky_factorization(sparse_matrix, ordering_method=ordering_method)
        
        if L_ordered is not None:
            plt.figure(figsize=(8, 8))
            plt.spy(L_ordered, markersize=0.5)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.title(f'Cholesky Factor L - {ordering_method.upper()} Ordering')
            plt.tight_layout(pad=0)
            out_path = os.path.join(output_dir, os.path.basename(mtx_file).replace('.mtx', f'_cholesky_{ordering_method}_ordering.png'))
            plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f'Saved Cholesky factor ({ordering_method.upper()} ordering) to {out_path}')
            
            factor_sparsity_ordered = (1 - L_ordered.count_nonzero() / (L_ordered.shape[0] * L_ordered.shape[1])) * 100
            print(f'Results:')
            print(f'  Factor non-zeros: {L_ordered.count_nonzero():,}')
            print(f'  Factor sparsity: {factor_sparsity_ordered:.2f}%')
            
            # Compare fill-in
            if L_natural is not None:
                original_nnz = sparse_matrix.count_nonzero()
                fill_in_natural = L_natural.count_nonzero() - original_nnz
                fill_in_ordered = L_ordered.count_nonzero() - original_nnz
                print(f'\n{"-"*40}')
                print('FILL-IN COMPARISON')
                print(f'{"-"*40}')
                print(f'Original matrix non-zeros: {original_nnz:,}')
                print(f'Natural ordering fill-in: {fill_in_natural:,} new non-zeros')
                print(f'{ordering_method.upper()} ordering fill-in: {fill_in_ordered:,} new non-zeros')
                if fill_in_natural > 0:
                    reduction = fill_in_natural - fill_in_ordered
                    reduction_pct = (reduction / fill_in_natural) * 100
                    print(f'Fill-in reduction: {reduction:,} ({reduction_pct:.1f}%)')
        
    except Exception as e:
        print(f"Error processing {mtx_file}: {e}")
        print(f"Error type: {type(e).__name__}")
        continue

print(f'\n{"="*60}')
print('Processing completed!')
print(f'{"="*60}')
