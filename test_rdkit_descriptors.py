import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs


def get_rdkit_descriptors(smiles):
    """
    Calculates all available RDKit descriptors for a given SMILES string.
    Returns: numpy array of descriptor values.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get the names of all available descriptors in RDKit
    # This includes 200+ properties like MolLogP, TPSAs, etc.
    desc_names = [desc[0] for desc in Descriptors._descList]

    # print(desc_names)

    # Initialize the calculator with all names
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    # Calculate descriptors
    desc_values = calculator.CalcDescriptors(mol)

    # Convert to float array and handle potential NaNs (e.g., from failed calculations)
    desc_array = np.array(desc_values, dtype=float)
    desc_array = np.nan_to_num(desc_array)  # Replace NaN with 0.0

    return desc_array.tolist()


def get_rdkit_fingerprints(smiles):

    mol = Chem.MolFromSmiles(smiles)

    # 1. RDKit (Topological) Fingerprints
    fpgen = AllChem.GetRDKitFPGenerator()
    fp_bitvect = fpgen.GetFingerprint(mol)

    # # 2. Morgan Fingerprints (Circular Fingerprints)
    # fpgen = AllChem.GetMorganGenerator()
    # fp_bitvect = fpgen.GetFingerprint(mol)

    # # 3. Atom Pairs and Topological Torsions
    # fpgen = AllChem.GetAtomPairGenerator()
    # fp_bitvect = fpgen.GetFingerprint(mol)

    # 4. Convert the RDKit BitVector object to a NumPy array
    fp_array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_bitvect, fp_array)

    return fp_array.tolist()


# Example usage for Glycine and Tryptophan
smiles_list = ["C(C(=O)O)N", "C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N"]
chem_features = [get_rdkit_descriptors(s) for s in smiles_list]
X_chem = np.vstack(chem_features)
fp_features = [get_rdkit_fingerprints(s)for s in smiles_list]
X_fp = np.vstack(fp_features)

print(f"The shape of chemical features extracted: {X_chem.shape}")
print(
    f"The [min, max] values of the 1st row of X_chem: {X_chem[:, 0].min()}, {X_chem[:, 0].max()}")
print(
    f"The [min, max] values of the 2nd row of X_chem: {X_chem[:, 1].min()}, {X_chem[:, 1].max()}")

print(f"The shape of fp features extracted: {X_fp.shape}")
print(
    f"The [min, max] values of the 1st row of X_fp: {X_fp[:, 0].min()}, {X_fp[:, 0].max()}")
print(
    f"The [min, max] values of the 2nd row of X_fp: {X_fp[:, 1].min()}, {X_fp[:, 1].max()}")
