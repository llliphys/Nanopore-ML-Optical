import numpy as np
from ase import Atoms
from ase.visualize import view

def flatten_to_xy_plane(molecule: Atoms):
    """
    Rotate a molecule so that its residual z-coordinates 
    (sum of squared distances to xy-plane) are minimized.
    
    Returns:
        rotated_molecule (ASE Atoms)
        rotation_matrix (3x3 np.ndarray)
    """

    # Extract coordinates
    coords = molecule.get_positions()

    # Center coordinates at centroid
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # Perform SVD / PCA
    # Covariance matrix
    cov = np.dot(centered.T, centered)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvectors by eigenvalues ascending (smallest variance → normal to plane)
    order = np.argsort(eigvals)
    principal_axes = eigvecs[:, order]

    # We want the smallest-variance axis to align to global z-axis = (0,0,1)
    # So rotation matrix is: principal_axes → identity
    R = principal_axes.T

    # Rotate the molecule
    rotated_coords = centered @ R.T

    # Move back to original centroid
    rotated_coords += centroid

    rotated_mol = molecule.copy()
    rotated_mol.set_positions(rotated_coords)

    return rotated_mol, R


# -------------------- TEST EXAMPLE -----------------------
if __name__ == "__main__":
    # Example water molecule
    mol = Atoms("H2O",
                positions=[
                    [0.0, 0.0, 0.1],
                    [0.8, 0.0, -0.1],
                    [-0.3, 0.6, -0.2]
                ])

    new_mol, R = flatten_to_xy_plane(mol)

    print("Rotation matrix:\n", R)
    print("\nOriginal z:", mol.get_positions()[:,2])
    print("Rotated  z:", new_mol.get_positions()[:,2])
    print("Sum of z² before:", np.sum(mol.get_positions()[:,2]**2))
    print("Sum of z² after :", np.sum(new_mol.get_positions()[:,2]**2))

    view(mol)

    view(new_mol)
