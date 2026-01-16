############################################################################################

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import ase.io
from ase import Atoms
from ase.visualize import view
from ase.calculators.siesta import Siesta
from ase.constraints import FixAtoms

import imageio
from PIL import Image

from dscribe.descriptors import SOAP

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from sklearn.preprocessing import OneHotEncoder


############################################################################################

os.environ["SIESTA_PP_PATH"] = "/home/llli/Potentials/SIESTA"
# os.environ["SIESTA_PP_PATH"] = "/home/Potentials/SIESTA"

############################################################################################

root_dir = os.getcwd()

if not os.path.exists(f"{root_dir}/INPUTS/structures"):
    os.makedirs(f"{root_dir}/INPUTS/structures")

structures_dir = f"{root_dir}/INPUTS/structures"

if not os.path.exists(f"{root_dir}/INPUTS/vaspposcars"):
    os.makedirs(f"{root_dir}/INPUTS/vaspposcars")

vaspposcars_dir = f"{root_dir}/INPUTS/vaspposcars"

if not os.path.exists(f"{root_dir}/INPUTS/siestafdfs"):
    os.makedirs(f"{root_dir}/INPUTS/siestafdfs")

siestafdfs_dir = f"{root_dir}/INPUTS/siestafdfs"

if not os.path.exists(f"{root_dir}/INPUTS/snapshots"):
    os.makedirs(f"{root_dir}/INPUTS/snapshots")

snapshots_dir = f"{root_dir}/INPUTS/snapshots"

if not os.path.exists(f"{root_dir}/INPUTS/animations"):
    os.makedirs(f"{root_dir}/INPUTS/animations")

animations_dir = f"{root_dir}/INPUTS/animations"

if not os.path.exists(f"{root_dir}/DFT/siestafdfs"):
    os.makedirs(f"{root_dir}/DFT/siestafdfs")

siestafdfs_dir = f"{root_dir}/DFT/siestafdfs"

if not os.path.exists(f"{root_dir}/NEGF/transiestafdfs"):
    os.makedirs(f"{root_dir}/NEGF/transiestafdfs")

transiestafdfs_dir = f"{root_dir}/NEGF/transiestafdfs"

if not os.path.exists(f"{root_dir}/ML/datasets"):
    os.makedirs(f"{root_dir}/ML/datasets")

mldatasets_dir = f"{root_dir}/ML/datasets"

if not os.path.exists(f"{root_dir}/TMP"):
    os.makedirs(f"{root_dir}/TMP")

tmp_dir = f"{root_dir}/TMP"

############################################################################################


def make_gif_with_pillow(image_folder, output_gif, duration=500):
    """
    Converts a list of images in a folder into a GIF animation and saves it.
    Args:
        image_folder (str): Path to the folder containing the images.
        output_path (str): Path where the GIF will be saved.
        duration (int): Duration of each frame in the GIF in milliseconds (default is 500 ms).
    """

    image_name = output_gif.split("/")[-1]

    # Step 1: Get a list of all image files in the specified folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(
        ('png', 'jpg', 'jpeg', 'bmp')) and image_name in f]

    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the specified folder.")
        return

    # Step 2: Sort the images by name to ensure they are in the correct order
    image_files.sort()

    # Step 3: Load the images into a list using PIL
    images = [Image.open(os.path.join(image_folder, image_file))
              for image_file in image_files]

    # Step 4: Convert the images to RGB format (necessary for GIF creation)
    images = [img.convert('RGB') for img in images]

    # Step 5: Create the GIF animation using the first image as the base and appending the rest
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,  # duration in milliseconds per frame
        loop=0
    )

    print(f"GIF animation created with pillow saved at: {output_gif}")

############################################################################################


def make_gif_with_imageio(image_folder, output_gif, duration=0.5):
    """
    Converts a list of images in a folder into a GIF animation and saves it.
    Args:
        image_folder (str): Path to the folder containing the images.
        output_path (str): Path where the GIF will be saved.
        duration (int): Duration of each frame in the GIF in seconds (default is 0.5 s).
    """

    # Step 1: Get a list of all image files in the specified folder
    image_files = [f for f in os.listdir(
        image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]

    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the specified folder.")
        return

    # Step 2: Sort the images by name to ensure they are in the correct order
    image_files.sort()

    images = []
    for image_file in image_files:
        images.append(imageio.imread(f"{image_folder}/{image_file}"))

    # Step 3: Create the GIF animation using the first image as the base and appending the rest
    # duration in seconds per frame
    imageio.mimsave(output_gif, images, duration=duration)

    print(f"GIF animation created with imageio saved at: {output_gif}")

############################################################################################


def make_fdf_name(file_name="POSCAR", run_task="relax", vdw_type=None):

    dft_task_list = ["NOSE", "SGLPT", "RELAX",
                     "SCF", "PDOS", "BAND", "WAVE", "OPTIC"]
    negf_task_list = ["LEAD", "ELEC", "DEVICE"]

    # print(file_name)

    if run_task.upper() in dft_task_list:
        for tag in ["POSCAR", "CONTCAR"]:
            if tag in file_name.upper():
                fdf_name = file_name.replace(tag, run_task.upper())
                break
            else:
                fdf_name = run_task.upper()
    elif run_task.upper() in negf_task_list:
        for tag in ["POSCAR", "CONTCAR"]:
            if tag in file_name.upper():
                fdf_name = file_name.replace(tag, run_task.upper())
                break
            else:
                fdf_name = run_task.upper()
    else:
        print(f"{run_task.upper()} IS CURRENTLY NOT SUPPOTRED!")
        exit(1)

    if vdw_type is not None:
        return fdf_name + "_VDW_" + vdw_type.upper()
    else:
        return fdf_name

############################################################################################


def make_siesta_fdf(atoms, file_name="POSCAR", run_task="relax", kpts_grid="1 1 1", vdw_type="KBM", constr_shape="circle", cutoff=5, plot_constr=False):

    os.chdir(siestafdfs_dir)

    fdf_name = make_fdf_name(file_name=file_name, run_task=run_task)
    calc = Siesta(label=fdf_name, atomic_coord_format="xyz",
                  fdf_arguments={"SystemLabel": "siesta"})
    calc.write_input(atoms, properties=['energy'])
    os.system(
        f"(echo {run_task.upper()}; echo {kpts_grid}; echo {vdw_type.upper()}; echo {fdf_name}) | SiestaParams")
    make_geometry_constraint(atoms, cutoff=cutoff, fdf_name=fdf_name,
                             constr_shape=constr_shape, plot_constr=plot_constr)

    os.chdir(root_dir)


############################################################################################

# Check if a point is located inside a hexagon
def inside_hexagon(xypos, center, cutoff):
    """Checks if a 2D point is inside a hexagon."""
    x, y = xypos - center
    qx, qy = abs(x), abs(y)
    return (qy <= cutoff * np.sqrt(3) / 2) and (qx <= cutoff - qy / np.sqrt(3))

############################################################################################


def make_geometry_constraint(atoms, cutoff=10, fdf_name="siesta", constr_shape="circle", plot_constr=False):

    xypos = atoms.positions[:, :2].round(decimals=8)

    xpos = atoms.positions[:, 0].round(decimals=8)
    ypos = atoms.positions[:, 1].round(decimals=8)

    xmid = (xpos.min() + xpos.max()) / 2
    ymid = (ypos.min() + ypos.max()) / 2
    center = np.array([xmid, ymid])

    # if "GRAPHENE" in fdf_name.upper():
    #     xpos_sort = np.sort(np.unique(xpos))[1::]
    #     ypos_sort = np.sort(np.unique(ypos))[0::]
    # if "MOS2_2H" in fdf_name.upper():
    #     xpos_sort = np.sort(np.unique(xpos))[1::]
    #     ypos_sort = np.sort(np.unique(ypos))[1::]

    # xmid = (xpos_sort.min() + xpos_sort.max()) / 2
    # ymid = (ypos_sort.min() + ypos_sort.max()) / 2
    # center = np.array([xmid, ymid])

    # use circle for geometry constraint
    if constr_shape.lower() == "circle":
        radius_to_center = np.linalg.norm(xypos - center, axis=1)
        relax_indices = np.where(radius_to_center <= cutoff)[0]
        constr_indices = [
            atom.index for atom in atoms if atom.index not in relax_indices]

    # use hexagon for geometry constraint
    if constr_shape.lower() == "hexagon":
        relax_indices = [atom.index for atom in atoms if inside_hexagon(
            atom.position[:2], center, cutoff)]
        constr_indices = [
            atom.index for atom in atoms if atom.index not in relax_indices]

    batch_size = 10
    os.system(f"echo '                    ' >> {fdf_name}.fdf")
    os.system(f"echo '###Geometry Constraint###' >> {fdf_name}.fdf")
    os.system(f"echo '%block Geometry.Constraints' >> {fdf_name}.fdf")
    if len(constr_indices) > batch_size:
        for i in range(0, len(constr_indices), batch_size):
            # 0-based --> 1-based
            indices = [idx+1 for idx in constr_indices[i:i+batch_size]]
            indices = ", ".join(map(str, indices))
            os.system(f"echo 'atom [{indices}]' >> {fdf_name}.fdf")
    else:
        indices = ", ".join([str(idx+1)
                            for idx in constr_indices])  # 0-based --> 1-based
        os.system(f"echo 'atom [{indices}]' >> {fdf_name}.fdf")
    os.system(f"echo '%endblock Geometry.Constraints' >> {fdf_name}.fdf")

    if plot_constr:
        plt.figure(figsize=(5, 5))
        plt.scatter(xpos[constr_indices], ypos[constr_indices], s=90, c="C1")
        plt.scatter(xpos[relax_indices], ypos[relax_indices], s=90, c="C2")
        plt.xlim(xpos.min()-1, xpos.max()+1)
        plt.ylim(ypos.min()-1, ypos.max()+1)
        title = "Constrain Relax"
        left, right = title.split()
        plt.text(0.65, 1.05, right, fontsize=25, color="C2",
                 ha='left', transform=plt.gca().transAxes)
        # plt.text(0.50, 1.05, " %s " % center, fontsize=25, color="k", ha='center', transform=plt.gca().transAxes)
        plt.text(0.45, 1.05, left, fontsize=25, color="C1",
                 ha='right', transform=plt.gca().transAxes)
        plt.show()


############################################################################################

def rotate_shift_molecule(surface, molecule, interdist, theta, phi, psi, center_type):

    if surface is not None:
        # print("surface.positions.shape:", surface.get_positions().shape)
        # print("surface.symbols.shape:", len(surface.get_chemical_symbols()))
        surface_center = surface.cell.sum(axis=0) / 2
    else:
        print("Surface Atoms Not Present!")
        exit(1)

    if molecule is not None:
        # print("molecule.positions.shape:", molecule.positions.shape)
        # print("molecule.symbols.shape:", len(molecule.get_chemical_symbols()))
        molecule.rotate(theta, "x", center="COM")
        molecule.rotate(phi, "y", center="COM")
        molecule.rotate(psi, "z", center="COM")
        molecule.center(axis=(0, 1, 2), about=surface_center)
    else:
        print("Molecule Atoms Not Present!")
        exit(1)

    if abs(interdist) > 1e-3:
        max_z_surface = max(surface.positions[:, 2])
        min_z_molecule = min(molecule.positions[:, 2])
        current_separation = min_z_molecule - max_z_surface
        dz = interdist - current_separation
        molecule.positions[:, 2] += dz

    return molecule

############################################################################################


def combine_surface_molecule(surface, molecule, distance, theta=0, phi=0, psi=0, center_type="COM"):

    # if surface is not None:
    #     surface_center =  surface.cell.sum(axis=0) / 2
    # else:
    #     print("Surface Atoms Not Present!")
    #     exit(1)

    # if molecule is not None:
    #     molecule.rotate(theta, "x", center="COM")
    #     molecule.rotate(phi, "y", center="COM")
    #     molecule.rotate(psi, "z", center="COM")
    #     molecule.center(axis=(0,1,2), about=surface_center)
    # else:
    #     print("Molecule Atoms Not Present!")
    #     exit(1)

    # if abs(distance) > 1e-3:
    #     max_z_surface = max(surface.positions[:, 2])
    #     min_z_molecule = min(molecule.positions[:, 2])
    #     current_separation = min_z_molecule - max_z_surface
    #     dz = distance - current_separation
    #     molecule.positions[:, 2] += dz

    molecule = rotate_shift_molecule(
        surface, molecule, distance, theta, phi, psi, center_type)

    combined = surface + molecule
    combined.center(vacuum=10, axis=2)
    combined.center(axis=(0, 1, 2))

    return combined

############################################################################################


def make_combined_name(surface_poscar, molecule_poscar, theta=None, phi=None, psi=None):

    surface_name = surface_poscar.upper()
    molecule_name = molecule_poscar.upper()

    angle_string = ""
    if theta is not None:
        angle_string += f"theta_{theta:03d}"
    if phi is not None:
        angle_string += f"_phi_{phi:03d}"
    if psi is not None:
        angle_string += f"_psi_{psi:03d}"

    combined_name = surface_name + "_" + molecule_name

    for tag in ["POSCAR_", "CONTCAR_"]:
        if tag in surface_name and tag in molecule_name:
            combined_name = surface_name.replace(
                tag, "") + "_" + molecule_name.replace(tag, "")

    if not angle_string:
        combined_name = "POSCAR" + "_" + combined_name.upper()
    else:
        combined_name = "POSCAR" + "_" + combined_name.upper() + "_" + \
            angle_string.upper()

    return combined_name

############################################################################################


def compute_atomic_pairs_old(atoms, theta=0, phi=0, psi=0, cutoff=5, n_select=10):

    xypos = atoms.positions[:, :2].round(decimals=8)

    xpos = atoms.positions[:, 0].round(decimals=8)
    ypos = atoms.positions[:, 1].round(decimals=8)

    xmid = (xpos.min() + xpos.max()) / 2
    ymid = (ypos.min() + ypos.max()) / 2
    center = np.array([xmid, ymid])

    radius_to_center = np.linalg.norm(xypos - center, axis=1)
    select_indices = np.where(radius_to_center <= cutoff)[0]

    atom_pair_indices = []
    atom_pair_symbols = []
    atom_pair_distances = []
    atom_pair_cosines = []

    for i in select_indices:
        for j in select_indices:

            if abs(i-j) != 0:

                ss = atoms[i].symbol
                ms = atoms[j].symbol
                vector = atoms[i].position - atoms[j].position
                distance = np.linalg.norm(vector)
                cosine = vector / (distance + 1e-6)

                atom_pair_indices.append([i, j])
                atom_pair_symbols.append([ss, ms])
                atom_pair_distances.append(distance)
                atom_pair_cosines.append(cosine)

    atom_pair_indices = np.stack(atom_pair_indices)
    atom_pair_symbols = np.stack(atom_pair_symbols)
    atom_pair_distances = np.array(atom_pair_distances)
    atom_pair_cosines = np.stack(atom_pair_cosines)

    atom_pair_distances = atom_pair_distances.round(decimals=8)
    atom_pair_cosines = atom_pair_cosines.round(decimals=8)

    atom_pair_distances[np.abs(atom_pair_distances) < 1e-6] = 0
    atom_pair_cosines[np.abs(atom_pair_cosines) < 1e-6] = 0

    sorted_indices = np.argsort(atom_pair_distances)
    atom_pair_indices = atom_pair_indices[sorted_indices]
    atom_pair_symbols = atom_pair_symbols[sorted_indices]
    atom_pair_distances = atom_pair_distances[sorted_indices]
    atom_pair_cosines = atom_pair_cosines[sorted_indices]

    pair_data = {
        "theta": theta,
        "phi": phi,
        "psi": psi,
    }

    if n_select is None:
        n_select = len(select_indices)

    for k in range(n_select):
        pair_data[f"dd{k}"] = atom_pair_distances[k]
    for k in range(n_select):
        pair_data[f"lx{k}"] = atom_pair_cosines[k, 0]
        pair_data[f"ly{k}"] = atom_pair_cosines[k, 1]
        pair_data[f"lz{k}"] = atom_pair_cosines[k, 2]
    for k in range(n_select):
        pair_data[f"si{k}"] = atom_pair_symbols[k, 0] + \
            f"{atom_pair_indices[k, 0]}"
        pair_data[f"mj{k}"] = atom_pair_symbols[k, 1] + \
            f"{atom_pair_indices[k, 1]}"

    # pair_data = []
    # for i in select_indices:
    #     for j in select_indices:
    #         if abs(i-j) != 0:
    #             dij = np.linalg.norm(atoms[i].position - atoms[j].position)
    #             pair_data.append(
    #                 {
    #                     # "i": int(i),
    #                     # "j": int(j),
    #                     "si": atoms[i].symbol+f"{i}", # surface atom symbol (s) + indiex (i)
    #                     "mj": atoms[j].symbol+f"{j}", # molecule atom symbol (m) + indiex (j)
    #                     "dd": dij,
    #                 }
    #             )
    #             # print(f"{atoms[i].symbol}{i} - {atoms[j].symbol}{j}: {dij:.3f}")

    return pair_data


############################################################################################

def compute_atomic_pairs(surface, molecule, distance, theta=0, phi=0, psi=0, center_type="COM", cutoff=5, n_select=10):

    molecule = rotate_shift_molecule(
        surface, molecule, distance, theta, phi, psi, center_type)

    atoms = surface + molecule
    atoms.center(vacuum=10, axis=2)
    atoms.center(axis=(0, 1, 2))

    xypos = atoms.positions[:, :2].round(decimals=8)

    xpos = atoms.positions[:, 0].round(decimals=8)
    ypos = atoms.positions[:, 1].round(decimals=8)

    xmid = (xpos.min() + xpos.max()) / 2
    ymid = (ypos.min() + ypos.max()) / 2
    center = np.array([xmid, ymid])

    # print("xypos.shape = ", xypos.shape)
    radii_to_center = np.linalg.norm(xypos - center, axis=1)
    select_indices = np.where(radii_to_center <= cutoff)[0]
    # print("radii_to_center.shape = ", radii_to_center.shape)

    atom_pair_indices = []
    atom_pair_symbols = []
    atom_pair_distances = []
    atom_pair_cosines = []

    for i in select_indices:
        for j in select_indices:

            if abs(i-j) != 0:

                ss = atoms[i].symbol
                ms = atoms[j].symbol
                vector = atoms[i].position - atoms[j].position
                distance = np.linalg.norm(vector)
                cosine = vector / (distance + 1e-6)

                atom_pair_indices.append([i, j])
                atom_pair_symbols.append([ss, ms])
                atom_pair_distances.append(distance)
                atom_pair_cosines.append(cosine)

    atom_pair_indices = np.stack(atom_pair_indices)
    atom_pair_symbols = np.stack(atom_pair_symbols)
    atom_pair_distances = np.array(atom_pair_distances)
    atom_pair_cosines = np.stack(atom_pair_cosines)

    atom_pair_distances = atom_pair_distances.round(decimals=8)
    atom_pair_cosines = atom_pair_cosines.round(decimals=8)

    atom_pair_distances[np.abs(atom_pair_distances) < 1e-6] = 0
    atom_pair_cosines[np.abs(atom_pair_cosines) < 1e-6] = 0

    sorted_indices = np.argsort(atom_pair_distances)
    atom_pair_indices = atom_pair_indices[sorted_indices]
    atom_pair_symbols = atom_pair_symbols[sorted_indices]
    atom_pair_distances = atom_pair_distances[sorted_indices]
    atom_pair_cosines = atom_pair_cosines[sorted_indices]

    pair_data = {
        "theta": theta,
        "phi": phi,
        "psi": psi,
    }

    if n_select is None:
        n_select = len(select_indices)

    for k in range(n_select):
        pair_data[f"dd{k}"] = atom_pair_distances[k]
    for k in range(n_select):
        pair_data[f"lx{k}"] = atom_pair_cosines[k, 0]
        pair_data[f"ly{k}"] = atom_pair_cosines[k, 1]
        pair_data[f"lz{k}"] = atom_pair_cosines[k, 2]
    for k in range(n_select):
        pair_data[f"si{k}"] = atom_pair_symbols[k, 0] + \
            f"{atom_pair_indices[k, 0]}"
        pair_data[f"mj{k}"] = atom_pair_symbols[k, 1] + \
            f"{atom_pair_indices[k, 1]}"

    return pair_data

############################################################################################


def compute_structural_features(surface, molecule, distance, theta=0, phi=0, psi=0, center_type="COM", n_select=10):
    """
    Efficient vectorized computation of distances + directional cosines
    between two ASE atoms objects: surface and molecule.
    Saves a globally sorted CSV file.

    Output columns:
        surface_index, molecule_index,
        surface_symbol, molecule_symbol,
        dx, dy, dz, distance,
        cosx, cosy, cosz
    """

    molecule = rotate_shift_molecule(
        surface, molecule, distance, theta, phi, psi, center_type)

    surface_positions = surface.get_positions()
    surface_symbols = surface.get_chemical_symbols()
    molecule_positions = molecule.get_positions()
    molecule_symbols = molecule.get_chemical_symbols()

    assert len(surface_positions) == len(surface_symbols)
    assert len(molecule_positions) == len(molecule_symbols)

    atom_pair_indices = []
    atom_pair_symbols = []
    atom_pair_distances = []
    atom_pair_cosines = []

    for i, surf_pos in enumerate(surface_positions):
        for j, mol_pos in enumerate(molecule_positions):

            ss = surface_symbols[i]
            ms = molecule_symbols[j]
            vector = surf_pos - mol_pos
            distance = np.linalg.norm(vector)
            cosine = vector / (distance + 1e-6)

            atom_pair_indices.append([i, j])
            atom_pair_symbols.append([ss, ms])
            atom_pair_distances.append(distance)
            atom_pair_cosines.append(cosine)

    atom_pair_indices = np.stack(atom_pair_indices)
    atom_pair_symbols = np.stack(atom_pair_symbols)
    atom_pair_distances = np.array(atom_pair_distances)
    atom_pair_cosines = np.stack(atom_pair_cosines)

    atom_pair_distances = atom_pair_distances.round(decimals=8)
    atom_pair_cosines = atom_pair_cosines.round(decimals=8)

    sorted_indices = np.argsort(atom_pair_distances)
    atom_pair_indices = atom_pair_indices[sorted_indices]
    atom_pair_symbols = atom_pair_symbols[sorted_indices]
    atom_pair_distances = atom_pair_distances[sorted_indices]
    atom_pair_cosines = atom_pair_cosines[sorted_indices]

    # ------------------------------------------------------------
    # Flatten all matrices for DataFrame
    # ------------------------------------------------------------

    data = {
        "theta": theta,
        "phi": phi,
        "psi": psi,
    }

    for k in range(n_select):
        data[f"dd{k}"] = atom_pair_distances[k]
    for k in range(n_select):
        data[f"lx{k}"] = atom_pair_cosines[k, 0]
        data[f"ly{k}"] = atom_pair_cosines[k, 1]
        data[f"lz{k}"] = atom_pair_cosines[k, 2]
    for k in range(n_select):
        data[f"si{k}"] = atom_pair_symbols[k, 0]+f"{atom_pair_indices[k, 0]}"
        data[f"mj{k}"] = atom_pair_symbols[k, 1]+f"{atom_pair_indices[k, 1]}"

    return data

############################################################################################


def scan_rotation_angles(surface, molecule, distance, rotate_angles, center_type="COM", n_select=10, csv_file="features.csv"):
    """Runs all rotations, computes pair data, selects smallest n distances."""

    feature_rows = []

    for theta, phi, psi in rotate_angles:

        data = compute_atomic_pairs(
            surface, molecule, distance, theta, phi, psi, center_type, n_select)

        feature_rows.append(data)

    df_raw = pd.DataFrame(feature_rows)
    df_raw.to_csv(f"{mldatasets_dir}/features_raw.csv", index=False)
    print(f"Saved {df_raw.shape[0]} samples to features_raw.csv")
    print(f"Saved {df_raw.shape[1]} features to features_raw.csv")

    # OneHot Encoder
    df = df_raw.copy()
    labels_for_ohe = ["si", "mj"]  # Atom-Pair Symbols (s, m) & Indices (i, j)
    feature_cols_for_ohe = [
        f"{l}{k}" for l in labels_for_ohe for k in range(n_select)]
    df_rest = df.drop(columns=feature_cols_for_ohe)

    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
    ohe.fit(df[feature_cols_for_ohe])  # perform encoding
    feature_labels = ohe.get_feature_names(feature_cols_for_ohe)
    feature_array = ohe.transform(
        df[feature_cols_for_ohe]).toarray()  # Transform the data
    df_ohe = pd.DataFrame(feature_array, columns=feature_labels)

    df_final = pd.concat([df_rest, df_ohe], axis=1)
    df_final.to_csv(f"{mldatasets_dir}/features_final.csv", index=False)
    print(f"Saved {df_final.shape[0]} samples to features_final.csv")
    print(f"Saved {df_final.shape[1]} features to features_final.csv")

############################################################################################


def compute_soap_features(atoms, cutoff=8, species=["H", "C", "O", "N"], periodic=False, r_cut=8, n_max=8, l_max=6, average="inner", n_jobs=1):

    xypos = atoms.positions[:, :2].round(decimals=8)

    xpos = atoms.positions[:, 0].round(decimals=8)
    ypos = atoms.positions[:, 1].round(decimals=8)

    xmid = (xpos.min() + xpos.max()) / 2
    ymid = (ypos.min() + ypos.max()) / 2
    center = np.array([xmid, ymid])

    radius_to_center = np.linalg.norm(xypos - center, axis=1)
    select_indices = np.where(radius_to_center <= cutoff)[0]

    atoms_select = atoms[select_indices]

    atoms = Atoms(atoms_select)

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
    )

    # Create the SOAP descriptor calculated for molecules
    soap_features = soap.create(atoms, n_jobs=n_jobs)

    # print(f"type of soap_features: {type(soap_features)}")
    # print(f"shape of soap_features: {soap_features.shape}")

    return soap_features.tolist()

############################################################################################


def compute_rdkit_features(atoms, fpSize=2048):

    atoms.write(f"{tmp_dir}/POSCAR.xyz")

    mol = Chem.MolFromXYZFile(f"{tmp_dir}/POSCAR.xyz")

    # 1. RDKit (Topological) Fingerprints
    fpgen = AllChem.GetRDKitFPGenerator(fpSize=fpSize)
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


############################################################################################


def compute_distance_features(surface, molecule, interdist=0, theta=0, phi=0, psi=0, center_type="COM", cutoff=10):

    edge_atoms = ["H", "C"]
    mole_atoms = ["H", "C", "O", "N"]

    xypos = surface.positions[:, :2].round(decimals=8)

    xpos = surface.positions[:, 0].round(decimals=8)
    ypos = surface.positions[:, 1].round(decimals=8)

    xmid = (xpos.min() + xpos.max()) / 2
    ymid = (ypos.min() + ypos.max()) / 2
    center = np.array([xmid, ymid])

    radius_to_center = np.linalg.norm(xypos - center, axis=1)
    select_indices = np.where(radius_to_center <= cutoff)[0]

    edge_positions = surface.positions[select_indices, :]
    edge_symbols = surface.symbols[select_indices]

    molecule = rotate_shift_molecule(
        surface, molecule, interdist, theta, phi, psi, center_type)

    mole_positions = molecule.positions
    mole_symbols = molecule.symbols

    min_distance_features = {}
    min_distances = []

    for edge_atom in edge_atoms:

        edge_positions_select = edge_positions[edge_symbols == edge_atom]

        print("-----------------------------")
        print(f"edge symbol = {edge_atom}")
        print(f"number of edge atoms = {len(edge_positions_select)}")

        for mole_atom in mole_atoms:

            mole_positions_select = mole_positions[mole_symbols == mole_atom]

            print(f"mole symbol = {mole_atom}")
            print(f"number of mole atoms = {len(mole_positions_select)}")

            # Calculate all pairwise Euclidean distances (L2 norm)
            distance_matrix = cdist(
                mole_positions_select, edge_positions_select, metric="euclidean")

            # Find the minimum distance in the entire matrix
            min_distance = np.min(distance_matrix)

            # Alternatively, find the minimum distance from each point in A to any in B
            # min_distances_mole2edge = np.min(distance_matrix, axis=1)

            # print(f"Number of atoms in molecule: {len(molecule)}")
            # print(f"minimum distances from mole {mole_atom} to edge {edge_atom}: {min_distances_mole2edge}")
            # print(f"Distance Matrix:\n{distance_matrix}")
            print(
                f"Min distances from {mole_atom} to {edge_atom}: {min_distance}")

            min_distances.append(min_distance)

            min_distance_features[f"Mole({mole_atom})-Edge({edge_atom})"] = min_distance

    # min_distances_array = np.array(min_distances_array)

    # return np.hstack(min_distances).tolist() if np.array(min_distances).shape[0] > 1 else min_distances
    return min_distances, min_distance_features


def make_data_frames(feature_rows, n_select=10, csv_file="features.csv"):
    """Runs all rotations, computes pair data, selects smallest n distances."""

    df_raw = pd.DataFrame(feature_rows)
    df_raw.to_csv(f"{mldatasets_dir}/features_raw.csv", index=False)
    print(f"Saved {df_raw.shape[0]} samples to features_raw.csv")
    print(f"Saved {df_raw.shape[1]} features to features_raw.csv")

    # # OneHot Encoder
    # df = df_raw.copy()
    # labels_for_ohe = ["si", "mj"]  # Atom-Pair Symbols (s, m) & Indices (i, j)
    # feature_cols_for_ohe = [
    #     f"{l}{k}" for l in labels_for_ohe for k in range(n_select)]
    # df_rest = df.drop(columns=feature_cols_for_ohe)

    # ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
    # ohe.fit(df[feature_cols_for_ohe])  # perform encoding
    # feature_labels = ohe.get_feature_names(feature_cols_for_ohe)
    # feature_array = ohe.transform(
    #     df[feature_cols_for_ohe]).toarray()  # Transform the data
    # df_ohe = pd.DataFrame(feature_array, columns=feature_labels)

    # df_final = pd.concat([df_rest, df_ohe], axis=1)
    # df_final.to_csv(f"{mldatasets_dir}/features_final.csv", index=False)
    # print(f"Saved {df_final.shape[0]} samples to features_final.csv")
    # print(f"Saved {df_final.shape[1]} features to features_final.csv")


############################################################################################

def main():

    parser = argparse.ArgumentParser(
        description="Process different rotations of a molecule w.r.t a surface.")

    parser.add_argument("--surface_poscar", type=str, default="CONTCAR_GRAPHENE_NANOPORE",
                        help="Surface (2D material) POSCAR or CONTCAR file")
    parser.add_argument("--molecule_poscar", type=str, default="CONTCAR_BIOMOL_ALA",
                        help="Molecule (e.g., amino acid) POSCAR or CONTCAR file")
    parser.add_argument("--inter_distance", type=float, default=0,
                        help="Vertical distance between surface and molecule (Ang)")
    parser.add_argument("--rotate_about_x", action="store_true",
                        default=True, help="Rotate about x-axis")
    parser.add_argument("--rotate_about_y", action="store_true",
                        default=True, help="Rotate about y-axis")
    parser.add_argument("--rotate_about_z", action="store_true",
                        default=True, help="Rotate about z-axis")
    parser.add_argument("--angle_increment", type=int,
                        default=90, help="Angle increment in units of degree")
    parser.add_argument("--num_rotations", type=int,
                        default=10, help="Number of rotation samplings")
    parser.add_argument("--view_structure", action="store_true",
                        default=False, help="Use ase gui to view structure")

    parser.add_argument("--soap_species", nargs="+", type=str,
                        default=["H", "C", "O", "N"], help="Species as a list of chemical symbols used by SOAP descriptor")
    parser.add_argument("--soap_r_cut", type=float,
                        default=8, help="A cutoff for local region in angstroms used by SOAP descriptor")
    parser.add_argument("--soap_n_max", type=int,
                        default=8, help="The number of radial basis functions used by SOAP descriptor")
    parser.add_argument("--soap_l_max", type=int,
                        default=6, help="The maximum degree of spherical harmonics used by SOAP descriptor")

    args = parser.parse_args()

    surface_poscar = args.surface_poscar
    molecule_poscar = args.molecule_poscar
    inter_distance = args.inter_distance
    rotate_about_x = args.rotate_about_x
    rotate_about_y = args.rotate_about_y
    rotate_about_z = args.rotate_about_z
    angle_increment = args.angle_increment
    num_rotations = args.num_rotations
    view_structure = args.view_structure
    soap_species = args.soap_species
    soap_n_max = args.soap_n_max
    soap_r_cut = args.soap_r_cut
    soap_l_max = args.soap_l_max

    # rotation about x-axis: theta angle: 0 - 360
    rotate_thetas = np.arange(0, 360, angle_increment)
    # rotation about y-axis: phi angle: 0 - 360
    rotate_phis = np.arange(0, 360, angle_increment)
    # rotation about z-axis: psi angle: 0 - 360
    rotate_psis = np.arange(0, 360, angle_increment)

   # rotate_thetas = np.linspace(0, 360, num_rotations, dtype=int)
   # rotate_phis = np.linspace(0, 360, num_rotations, dtype=int)
   # rotate_psis = np.linspace(0, 360, num_rotations, dtype=int)

    if rotate_about_x:
        rotate_angles = [(int(theta), 0, 0) for theta in rotate_thetas]
    if rotate_about_y:
        rotate_angles = [(0, int(phi), 0) for phi in rotate_phis]
    if rotate_about_z:
        rotate_angles = [(0, 0, int(psi)) for psi in rotate_psis]

    if rotate_about_x and rotate_about_y:
        rotate_angles = [(int(theta), int(phi), 0)
                         for theta in rotate_thetas for phi in rotate_phis]
    if rotate_about_x and rotate_about_z:
        rotate_angles = [(int(theta), 0, int(psi))
                         for theta in rotate_thetas for psi in rotate_psis]
    if rotate_about_y and rotate_about_z:
        rotate_angles = [(0, int(phi), int(psi))
                         for phi in rotate_phis for psi in rotate_psis]

    if rotate_about_x and rotate_about_y and rotate_about_z:
        rotate_angles = [(int(theta), int(phi), int(
            psi)) for theta in rotate_thetas for phi in rotate_phis for psi in rotate_psis]

    surface_atoms = ase.io.read(f"{structures_dir}/{surface_poscar}")
    molecule_atoms = ase.io.read(f"{structures_dir}/{molecule_poscar}")

    if view_structure:
        view(surface_atoms)
        view(molecule_atoms)

    features_rows = []
    for theta, phi, psi in rotate_angles:

        combined = combine_surface_molecule(
            surface_atoms, molecule_atoms, inter_distance, theta=theta, phi=phi, psi=psi)

        combined_name = make_combined_name(
            surface_poscar, molecule_poscar, theta=theta, phi=phi, psi=psi)

        combined.write(f"{vaspposcars_dir}/{combined_name}.vasp",
                       format="vasp", vasp5=True, direct=True, sort=True)

        combined.write(f"{snapshots_dir}/{combined_name}.png",  format="png")

        # make_siesta_fdf(combined, file_name=f"{combined_name}", run_task="relax", cutoff=8.5, plot_constr=True)

        # pair_data = compute_atomic_pairs(combined, theta=theta, phi=phi, psi=psi, cutoff=8)

        # features_row = compute_soap_features(combined, cutoff=8, species=soap_species,
        #                                      r_cut=soap_r_cut, n_max=soap_n_max, l_max=soap_l_max, n_jobs=2)

        # print(min(features_row), max(features_row))

        # features_row = np.concatenate(
        #     [np.array([np.cos(theta), np.cos(phi), np.cos(psi)]), np.array(features_row)])

        # features_rows.append(features_row)

        min_distances, min_distances_features = compute_distance_features(
            surface_atoms, molecule_atoms, interdist=inter_distance, theta=theta, phi=phi, psi=psi, cutoff=10)

        print(min_distances)
        print(min_distances_features)

        features_rows.append(min_distances_features)

    # Create GIF animation if the number of images is less than 1000
    # if len(rotate_angles) < 1000:
    #     output_gif_name = make_combined_name(surface_poscar, molecule_poscar)
    #     make_gif_with_pillow(image_folder=f"{snapshots_dir}", output_gif=f"{animations_dir}/{output_gif_name}.gif", duration=500)

    # scan_rotation_angles(surface_atoms, molecule_atoms, inter_distance, rotate_angles, center_type="COM", csv_file="features.csv")

    make_data_frames(features_rows)

############################################################################################


if __name__ == "__main__":
    main()
