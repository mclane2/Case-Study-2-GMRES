import numpy as np
import ssgetpy
import scipy.sparse as sp
import scipy.io

def arnoldi_iteration(A, V, H, j):
    """
    Perform one step of the Arnoldi iteration.
    
    Takes the existing basis V and Hessenberg matrix H,
    and extends them by one vector/column.
    
    Returns:
        breakdown : True if h_{j+1,j} is essentially zero (lucky breakdown)
    """
    w = A @ V[:, j]

    # Orthogonalise against all previous basis vectors
    for i in range(j + 1):
        H[i, j] = np.dot(w, V[:, i])
        w = w - H[i, j] * V[:, i]

    H[j + 1, j] = np.linalg.norm(w)

    if H[j + 1, j] < 1e-14:
        return True  # Code terminates algorithm if true

    V[:, j + 1] = w / H[j + 1, j]
    return False


def givens_rotation(a, b):
    """
    Compute the Givens rotation that zeros out b:
    
        [ cs  sn ] [ a ]   [ r ]
        [-sn  cs ] [ b ] = [ 0 ]
    
    Returns cs, sn.
    """
    if abs(b) < 1e-15:
        return 1.0, 0.0
    elif abs(b) > abs(a):
        tau = -a / b
        sn = 1.0 / np.sqrt(1 + tau**2)
        cs = sn * tau
    else:
        tau = -b / a
        cs = 1.0 / np.sqrt(1 + tau**2)
        sn = cs * tau
    return cs, sn


def apply_givens(cs, sn, h_i, h_ip1):
    """
    Apply a Givens rotation to a pair of entries:
    
        [ cs  sn ] [ h_i   ]   [ new_h_i   ]
        [-sn  cs ] [ h_ip1 ] = [ new_h_ip1 ]
    """
    new_hi   =  cs * h_i + sn * h_ip1
    new_hip1 = -sn * h_i + cs * h_ip1
    return new_hi, new_hip1


def fetch_matrix(group, name):
    """Download a SuiteSparse matrix and return it as a scipy sparse matrix."""
    import os
    import glob

    results = ssgetpy.search(name=name, group=group)
    if len(results) == 0:
        print(f"  WARNING: matrix {group}/{name} not found, skipping.")
        return None

    dest = os.path.join(os.getcwd(), "suitesparse_matrices")
    os.makedirs(dest, exist_ok=True)

    mat = results[0]
    mat.download(format='MM', destpath=dest)

    # ssgetpy downloads .tar.gz — extract any that exist
    import tarfile
    for tgz in glob.glob(os.path.join(dest, "*.tar.gz")):
        with tarfile.open(tgz, 'r:gz') as tar:
            tar.extractall(path=dest)

    # ssgetpy folder structure varies — just search for the .mtx file
    pattern = os.path.join(dest, "**", f"{name}.mtx")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        print(f"  WARNING: downloaded but can't find {name}.mtx in {dest}")
        print(f"  Contents: {os.listdir(dest)}")
        return None

    A = scipy.io.mmread(matches[0])
    return sp.csr_matrix(A)
    