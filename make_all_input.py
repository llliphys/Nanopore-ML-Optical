############################################################################################

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ase.io
from ase.visualize import view
from ase.calculators.siesta import Siesta
from ase.constraints import FixAtoms
import imageio
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

############################################################################################

os.environ["SIESTA_PP_PATH"] = "/home/llli/Potentials/SIESTA"
# os.environ["SIESTA_PP_PATH"] = "/home/Potentials/SIESTA"

############################################################################################
    
root_dir = os.getcwd()

if not os.path.exists(f"{root_dir}/INPUTS/structures"):
    os.makedirs(f"{root_dir}/INPUTS/structures")

structures_dir = f"{root_dir}/INPUTS/structures"

# if not os.path.exists(f"{root_dir}/INPUTS/rotations"):
#     os.makedirs(f"{root_dir}/INPUTS/rotations")

# rotations_dir = f"{root_dir}/INPUTS/rotations"

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
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp')) and image_name in f]

    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the specified folder.")
        return

    # Step 2: Sort the images by name to ensure they are in the correct order
    image_files.sort()

    # Step 3: Load the images into a list using PIL
    images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]

    # Step 4: Convert the images to RGB format (necessary for GIF creation)
    images = [img.convert('RGB') for img in images]

    # Step 5: Create the GIF animation using the first image as the base and appending the rest
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration, # duration in milliseconds per frame
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
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]

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
    imageio.mimsave(output_gif, images, duration=duration) # duration in seconds per frame

    print(f"GIF animation created with imageio saved at: {output_gif}")

############################################################################################

def make_fdf_name(file_name="POSCAR", run_task="relax", vdw_type=None):

    dft_task_list = ["NOSE", "SGLPT", "RELAX", "SCF", "PDOS", "BAND", "WAVE", "OPTIC"]
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
    calc = Siesta(label=fdf_name, atomic_coord_format="xyz", fdf_arguments={"SystemLabel": "siesta"})
    calc.write_input(atoms, properties=['energy'])
    os.system(f"(echo {run_task.upper()}; echo {kpts_grid}; echo {vdw_type.upper()}; echo {fdf_name}) | SiestaParams")
    make_geometry_constraint(atoms, cutoff=cutoff, fdf_name=fdf_name, constr_shape=constr_shape, plot_constr=plot_constr)

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

    if "GRAPHENE" in fdf_name.upper():
        xpos_sort = np.sort(np.unique(xpos))[1::]
        ypos_sort = np.sort(np.unique(ypos))[0::]
    if "MOS2_2H" in fdf_name.upper():
        xpos_sort = np.sort(np.unique(xpos))[1::]
        ypos_sort = np.sort(np.unique(ypos))[1::]
    xmid = (xpos_sort.min() + xpos_sort.max()) / 2
    ymid = (ypos_sort.min() + ypos_sort.max()) / 2
    center = np.array([xmid, ymid])

    # use circle for geometry constraint
    if constr_shape.lower() == "circle":
        radius_to_center = np.linalg.norm(xypos - center, axis=1)
        relax_indices = np.where(radius_to_center <= cutoff)[0]
        constr_indices = [atom.index for atom in atoms if atom.index not in relax_indices]

    # use hexagon for geometry constraint
    if constr_shape.lower() == "hexagon":
        relax_indices = [atom.index for atom in atoms if inside_hexagon(atom.position[:2], center, cutoff)]
        constr_indices = [atom.index for atom in atoms if atom.index not in relax_indices]

    batch_size = 10
    os.system(f"echo '                    ' >> {fdf_name}.fdf")
    os.system(f"echo '###Geometry Constraint###' >> {fdf_name}.fdf")
    os.system(f"echo '%block Geometry.Constraints' >> {fdf_name}.fdf")
    if len(constr_indices) > batch_size:
        for i in range(0, len(constr_indices), batch_size):
            indices = [idx+1 for idx in constr_indices[i:i+batch_size]] # 0-based --> 1-based
            indices = ", ".join(map(str, indices))
            os.system(f"echo 'atom [{indices}]' >> {fdf_name}.fdf")
    else:
        indices = ", ".join([str(idx+1) for idx in constr_indices]) # 0-based --> 1-based
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
        plt.text(0.65, 1.05, right, fontsize=25, color="C2", ha='left', transform=plt.gca().transAxes)
        # plt.text(0.50, 1.05, " %s " % center, fontsize=25, color="k", ha='center', transform=plt.gca().transAxes)
        plt.text(0.45, 1.05, left, fontsize=25, color="C1", ha='right', transform=plt.gca().transAxes)
        plt.show()


############################################################################################

def rotate_shift_molecule(surface, molecule, distance, theta, phi, psi, center_type):

    if surface is not None:
        # print("surface.positions.shape:", surface.get_positions().shape)
        # print("surface.symbols.shape:", len(surface.get_chemical_symbols()))
        surface_center =  surface.cell.sum(axis=0) / 2
    else:
        print("Surface Atoms Not Present!")
        exit(1)

    if molecule is not None:
        # print("molecule.positions.shape:", molecule.positions.shape)
        # print("molecule.symbols.shape:", len(molecule.get_chemical_symbols()))
        molecule.rotate(theta, "x", center="COM")
        molecule.rotate(phi, "y", center="COM")
        molecule.rotate(psi, "z", center="COM")
        molecule.center(axis=(0,1,2), about=surface_center)
    else:
        print("Molecule Atoms Not Present!")
        exit(1)

    if abs(distance) > 1e-3:
        max_z_surface = max(surface.positions[:, 2])
        min_z_molecule = min(molecule.positions[:, 2])
        current_separation = min_z_molecule - max_z_surface
        dz = distance - current_separation
        molecule.positions[:, 2] += dz

    return molecule

############################################################################################

# Modify the rotation function to center the molecule before rotation
def combine_surface_molecule(surface, molecule, distance, theta=0, phi=0, psi=0, center_type="COM"):
    """
    Rotate a molecule around its center of mass or geometrical center.

    Parameters:
    - molecule: ASE Atoms object
    - theta: Polar angle (in radians) around z-axis
    - phi: Azimuthal angle (in radians) around y-axis
    - center_type: 'COM' (center of mass) or 'geometrical' (geometrical center)

    Returns:
    - Rotated positions as a numpy array
    """
  
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

    molecule = rotate_shift_molecule(surface, molecule, distance, theta, phi, psi, center_type)

    combined = surface + molecule
    combined.center(vacuum=10, axis=2)
    combined.center(axis=(0,1,2))
    

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
            combined_name = surface_name.replace(tag, "") + "_" + molecule_name.replace(tag, "") 
    
    if not angle_string: 
        combined_name = "POSCAR" + "_" + combined_name.upper()
    else:
        combined_name = "POSCAR" + "_" + combined_name.upper() + "_" + angle_string.upper()

    return combined_name

############################################################################################

def compute_structural_features(surface, molecule, distance, theta=0, phi=0, psi=0, center_type="COM", n_select=50):
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

    molecule = rotate_shift_molecule(surface, molecule, distance, theta, phi, psi, center_type)
    
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

    # for k in range(n_select):
    #     data[f"ss{k}"] = atom_pair_symbols[k, 0]
    # for k in range(n_select):
    #     data[f"ms{k}"] = atom_pair_symbols[k, 1]
    # for k in range(n_select):
    #     data[f"si{k}"] = atom_pair_indices[k, 0]
    # for k in range(n_select):
    #     data[f"mi{k}"] = atom_pair_indices[k, 1]

    for k in range(n_select):
        data[f"si{k}"] = atom_pair_symbols[k, 0]+f"{atom_pair_indices[k, 0]}"
        data[f"mj{k}"] = atom_pair_symbols[k, 1]+f"{atom_pair_indices[k, 1]}"

        # data[f"si_{k}"] = int(atom_pair_indices[k, 0])
        # data[f"mi_{k}"] = int(atom_pair_indices[k, 1])
        # data[f"ss_{k}"] = atom_pair_symbols[k, 0]
        # data[f"ms_{k}"] = atom_pair_symbols[k, 1]
        # data[f"lx_{k}"] = atom_pair_cosines[k, 0]
        # data[f"ly_{k}"] = atom_pair_cosines[k, 1]
        # data[f"lz_{k}"] = atom_pair_cosines[k, 2]

    return data

############################################################################################


def scan_rotation_angles(surface, molecule, distance, rotate_angles, center_type="COM", n_select=50, csv_file="features.csv"):
    """Runs all rotations, computes pair data, selects smallest n distances."""

    feature_rows = []

    for theta, phi, psi in rotate_angles:
    
        data = compute_structural_features(surface, molecule, distance, theta, phi, psi, center_type, n_select)

        feature_rows.append(data)

    df = pd.DataFrame(feature_rows)
    df.to_csv(csv_file, index=False)
    # print(f"Saved {df.shape[1]} features to features.csv")

    # ohe = OneHotEncoder(categories='auto')
    # labels = ["ss", "ms"]
    # feature_cols = [f"{l}{k}" for l in labels for k in range(n_select)]
    # feature_arr = ohe.fit_transform(df[feature_cols]).toarray()
    # feature_labels = ohe.categories_
    # feature_labels = np.array(feature_labels).ravel()


    labels = ["si", "mj"]
    feature_cols = [f"{l}{k}" for l in labels for k in range(n_select)]

    df_tmp = df.drop(columns=feature_cols)

    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')

    # Perform encoding
    ohe.fit(df[feature_cols])

    # FIX: Use the correct method to get the new column names
    feature_labels = ohe.get_feature_names(feature_cols)

    # Transform the data
    feature_arr = ohe.transform(df[feature_cols]).toarray()

    df_ohe = pd.DataFrame(feature_arr, columns=feature_labels)
    df_ohe.to_csv("features_ohe.csv", index=False)

    # print(f"Saved {df_ohe.shape[0]} samples to features_ohe.csv")
    # print(f"Saved {df_ohe.shape[1]} features to features_ohe.csv")

    df_final = pd.concat([df_tmp, df_ohe], axis=1)
    df_final.to_csv("features_final.csv", index=False)

    print(f"Saved {df_final.shape[0]} samples to features_final.csv")
    print(f"Saved {df_final.shape[1]} features to features_final.csv")


############################################################################################

def main():

    parser = argparse.ArgumentParser(description="Process different rotations of a molecule w.r.t a surface.")

    parser.add_argument("--surface_poscar", help="Surface (2D material) POSCAR or CONTCAR file")
    parser.add_argument("--molecule_poscar", help="Molecule (e.g., amino acid) POSCAR or CONTCAR file")
    parser.add_argument("--inter_distance", type=float, default=0, help="Vertical distance between surface and molecule (Ang)")
    parser.add_argument("--rotate_about_x", action="store_true", default=True, help="Rotate about x-axis")
    parser.add_argument("--rotate_about_y", action="store_true", default=True, help="Rotate about y-axis")
    parser.add_argument("--rotate_about_z", action="store_true", default=True, help="Rotate about z-axis")
    parser.add_argument("--angle_increment", type=int, default=60, help="Angle increment in units of degree")
    # parser.add_argument("--num_rotations", type=int, default=60, help="Number of rotation samplings")
    parser.add_argument("--view_structure", action="store_true", default=False, help="Use ase gui to view structure")

    args = parser.parse_args()

    surface_poscar = args.surface_poscar
    molecule_poscar = args.molecule_poscar
    inter_distance = args.inter_distance
    rotate_about_x = args.rotate_about_x
    rotate_about_y = args.rotate_about_y
    rotate_about_z = args.rotate_about_z
    angle_increment = args.angle_increment
    # num_rotations = args.num_rotations
    view_structure = args.view_structure

    rotate_thetas = np.arange(0, 380, angle_increment)  # rotation about x-axis: theta angle: 0 - 360
    rotate_phis = np.arange(0, 380, angle_increment)    # rotation about y-axis: phi angle: 0 - 360
    rotate_psis = np.arange(0, 380, angle_increment)    # rotation about z-axis: psi angle: 0 - 360

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
        rotate_angles = [(int(theta), int(phi), 0) for theta in rotate_thetas for phi in rotate_phis]
    if rotate_about_x and rotate_about_z:
        rotate_angles = [(int(theta), 0, int(psi)) for theta in rotate_thetas for psi in rotate_psis]
    if rotate_about_y and rotate_about_z:
        rotate_angles = [(0, int(phi), int(psi)) for phi in rotate_phis for psi in rotate_psis]

    if rotate_about_x and rotate_about_y and rotate_about_z:
        rotate_angles = [(int(theta), int(phi), int(psi)) for theta in rotate_thetas for phi in rotate_phis for psi in rotate_psis]

    surface_atoms = ase.io.read(f"{structures_dir}/{surface_poscar}")
    molecule_atoms = ase.io.read(f"{structures_dir}/{molecule_poscar}")

    if view_structure:
        view(surface_atoms)
        view(molecule_atoms)

    for theta, phi, psi in rotate_angles:

        combined = combine_surface_molecule(surface_atoms, molecule_atoms, inter_distance, theta=theta, phi=phi, psi=psi)

        combined_name = make_combined_name(surface_poscar, molecule_poscar, theta=theta, phi=phi, psi=psi)
        
        print("Combined Name: ", combined_name)

        combined.write(f"{rotations_dir}/{combined_name}.vasp", format="vasp", vasp5=True, direct=True, sort=True)

        combined.write(f"{snapshots_dir}/{combined_name}.png",  format="png")
        
        make_siesta_fdf(combined, file_name=f"{combined_name}", run_task="relax", cutoff=8.5, plot_constr=True)

    # Create GIF animation if the number of images is less than 1000 
    # if len(rotate_angles) < 1000: 
    #     output_gif_name = make_combined_name(surface_poscar, molecule_poscar)
    #     make_gif_with_pillow(image_folder=f"{snapshots_dir}", output_gif=f"{animations_dir}/{output_gif_name}.gif", duration=500)

    # scan_rotation_angles(surface_atoms, molecule_atoms, inter_distance, rotate_angles, center_type="COM", n_select=20, csv_file="features.csv")

############################################################################################

if __name__ == "__main__":
    main()
