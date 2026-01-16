import ase
from ase import Atoms
from dscribe.descriptors import SOAP
from ase.build import molecule
import ase.io
import numpy as np

import os
# from pymatgen.core import Structure
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


def inside_hexagon(xypos, center, cutoff):
    """Checks if a 2D point is inside a hexagon."""
    x, y = xypos - center
    qx, qy = abs(x), abs(y)
    return (qy <= cutoff * np.sqrt(3) / 2) and (qx <= cutoff - qy / np.sqrt(3))


# Initial parameters to create SOAP descriptor
species = ["H", "C", "O", "N"]
r_cut = 10
n_max = 10
l_max = 10


# Molecules created as ASE Atoms Objects
water = molecule("H2O")
carbon_oxide = molecule("CO2")
molecules = [water, carbon_oxide]

methan = molecule("CH3")
peroxide = molecule("H2O2")
molecules = [methan, peroxide]

print(peroxide)

for average in ["off", "inner", "outer"]:

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
    )

    # Create the SOAP descriptor calculated for molecules
    soap_molecules = soap.create(molecules, n_jobs=2)
    # print(f"SOAP Average = {average}; SOAP.shape = ", soap_molecules.shape)

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        # average=average,
    )

    derivatives, _ = soap.derivatives(molecules, method="analytical")

    # print(f"SOAP Derivative Average = {average}; SOAP.Derivative.shape = ", derivatives.shape)

    # print(derivatives[0, 0, :, :, 0])


# Create the SOAP water descriptor calculated for specified center(s)
# soap_water_one_center = soap.create(water, centers=[0])
#
# print(soap_water_one_center.shape)
#
# print(soap_water_one_center)

# # Create output for multiple system
# samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]
# centers = [[0], [1, 2], [1, 2]]
# coulomb_matrices = soap.create(samples, centers)            # Serial
# coulomb_matrices = soap.create(samples, centers, n_jobs=2)  # Parallel

# Select a certain amount of atoms within the specified range
atoms = ase.io.read("POSCAR")

xpos = atoms.positions[:, 0].round(decimals=8)
ypos = atoms.positions[:, 1].round(decimals=8)
xmid = (xpos.min() + xpos.max()) / 2
ymid = (ypos.min() + ypos.max()) / 2
center = np.array([xmid, ymid])

atoms_select = [atom for atom in atoms if inside_hexagon(
    atom.position[:2], center=center, cutoff=6)]

atoms = Atoms(atoms_select)

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average='inner',
)

# Create the SOAP descriptor calculated for molecules
soap_atoms = soap.create(atoms, n_jobs=2)

print(f"type of soap_atoms: {type(soap_atoms)}")
print(f"shape of soap_atoms: {soap_atoms.shape}")

ase.io.read("POSCAR_ALA").write("ALA.xyz")

mol = Chem.MolFromXYZFile("ALA.xyz")

# rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
# ao = rdFingerprintGenerator.AdditionalOutput()
# fp_bitvect = rdkgen.GetFingerprint(mol, additionalOutput=ao)

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

print(f"type of fp_array: {type(fp_array)}")
print(f"shape of fp_array: {fp_array.shape}")

features = np.concatenate([soap_atoms, fp_array])
print(f"type of features: {type(features)}")
print(f"shape of features: {features.shape}")
