import os
import numpy as np
import ase
from ase.calculators.lj import LennardJones
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP

root_dir = os.getcwd()

if not os.path.exists(f"{root_dir}/DATASETS"):
    os.makedirs(f"{root_dir}/DATASETS")

# Setting up the SOAP descriptor
soap = SOAP(
    species=["H"],
    periodic=False,
    r_cut=8.0,
    sigma=0.5,
    n_max=8,
    l_max=3,
)

# Generate dataset of Lennard-Jones energies and forces
n_samples = 200
traj = []
n_atoms = 2
energies = np.zeros(n_samples)
forces = np.zeros((n_samples, n_atoms, 3))
r = np.linspace(2.5, 5.0, n_samples)
for i, d in enumerate(r):
    a = ase.Atoms('HH', positions = [[-0.5 * d, 0, 0], [0.5 * d, 0, 0]])
    a.set_calculator(LennardJones(epsilon=1.0 , sigma=2.9))
    traj.append(a)
    energies[i] = a.get_total_energy()
    forces[i, :, :] = a.get_forces()


# Create the SOAP desciptors and their derivatives for all samples. One center
# is chosen to be directly between the atoms.
derivatives, descriptors = soap.derivatives(
    traj,
    centers=[[[0, 0, 0]]] * len(r),
    method="analytical"
)

# Save to disk for later training
# if not os.path.isfile(f"{root_dir}/DATASETS/r.npy"):
np.save(f"{root_dir}/DATASETS/r.npy", r)
# if not os.path.isfile(f"{root_dir}/DATASETS/E.npy"):
np.save(f"{root_dir}/DATASETS/E.npy", energies)
# if not os.path.isfile(f"{root_dir}/DATASETS/D.npy"):
np.save(f"{root_dir}/DATASETS/D.npy", descriptors)
# if not os.path.isfile(f"{root_dir}/DATASETS/dD_dr.npy"):
np.save(f"{root_dir}/DATASETS/dD_dr.npy", derivatives)
# if not os.path.isfile(f"{root_dir}/DATASETS/F.npy"):
np.save(f"{root_dir}/DATASETS/F.npy", forces)


# Plot the energies to validate them
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
line, = ax.plot(r, energies)
plt.xlabel("Distance (Å)")
plt.ylabel("Energy (eV)")
plt.show()
