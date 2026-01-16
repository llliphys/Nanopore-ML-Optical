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
from ase.geometry.analysis import Analysis
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

if not os.path.exists(f"{root_dir}/INPUTS/pseudopots"):
    os.makedirs(f"{root_dir}/INPUTS/pseudopots")

pseudopots_dir = f"{root_dir}/INPUTS/pseudopots"

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
    image_files = [file for file in os.listdir(image_folder) if file.endswith(
        ('png', 'jpg', 'jpeg', 'bmp')) and image_name in file]

    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the specified folder.")
        # exit(1)
        return

    # Step 2: Sort the images by name to ensure they are in the correct order
    image_files.sort()

    # Step 3: Load the images into a list using PIL
    images = [Image.open(os.path.join(image_folder, image_file))
              for image_file in image_files]

    # Step 4: Convert the images to RGB format (necessary for GIF creation)
    images = [imgage.convert('RGB') for imgage in images]

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

    image_name = output_gif.split("/")[-1]

    # Step 1: Get a list of all image files in the specified folder
    image_files = [file for file in os.listdir(image_folder) if file.endswith(
        ('png', 'jpg', 'jpeg', 'bmp')) and image_name in file]

    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the specified folder.")
        # exit(1)
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

    if run_task.upper() in file_name.upper():
        file_name = file_name.replace(f"{run_task.upper()}_", "")

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


def make_siesta_fdf(atoms, file_name="POSCAR", run_task="relax", kpts_grid="1 1 1", vdw_type="KBM", make_constr=False, constr_shape="circle", cutoff=5, plot_constr=False):

    os.chdir(siestafdfs_dir)

    fdf_name = make_fdf_name(file_name=file_name, run_task=run_task)
    calc = Siesta(label=fdf_name, atomic_coord_format="xyz",
                  fdf_arguments={"SystemLabel": "siesta"})
    calc.write_input(atoms, properties=['energy'])
    os.system(
        f"(echo {run_task.upper()}; echo {kpts_grid}; echo {vdw_type.upper()}; echo {fdf_name}) | SiestaParams")

    if make_constr:
        make_geometry_constraint(atoms, cutoff=cutoff, fdf_name=fdf_name,
                                 constr_shape=constr_shape, plot_constr=plot_constr)

    os.chdir(root_dir)

############################################################################################


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
        # plt.figure(figsize=(5, 5))
        plt.figure()
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

    if surface is not None:
        surface_center = surface.cell.sum(axis=0) / 2
    else:
        print("Surface Atoms Not Present!")
        exit(1)

    if molecule is not None:
        molecule.rotate(theta, "x", center="COM")
        molecule.rotate(phi, "y", center="COM")
        molecule.rotate(psi, "z", center="COM")
        molecule.center(axis=(0, 1, 2), about=surface_center)
    else:
        print("Molecule Atoms Not Present!")
        exit(1)

    if abs(distance) > 1e-3:
        max_z_surface = max(surface.positions[:, 2])
        min_z_molecule = min(molecule.positions[:, 2])
        current_separation = min_z_molecule - max_z_surface
        dz = distance - current_separation
        molecule.positions[:, 2] += dz

    combined = surface + molecule
    # combined.center(vacuum=10, axis=2)
    # combined.center(axis=(0,1,2))

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

    # if not angle_string:
    #     combined_name = "POSCAR" + "_" + combined_name.upper()
    # else:
    #     combined_name = "POSCAR" + "_" + combined_name.upper() + "_" + angle_string.upper()

    if not angle_string:
        combined_name = combined_name.upper()
    else:
        combined_name = combined_name.upper() + "_" + angle_string.upper()

    return combined_name

############################################################################################


def set_lattice_vector(source_atoms, target_atoms, which_vector="c"):
    """
    Sets the lattice vector of the target_atoms
    to match that of the source_atoms.

    Parameters:
    - source_atoms (ase.Atoms): The source Atoms object.
    - target_atoms (ase.Atoms): The target Atoms object.
    """

    # Get the lattice vector c of the source_atoms
    source_cell = source_atoms.get_cell()

    # Get the cell of the target_atoms
    target_cell = target_atoms.get_cell()

    if which_vector.lower() == "a":
        a_vector = source_cell[0]
        target_cell[0] = a_vector
    if which_vector.lower() == "b":
        b_vector = source_cell[1]  # Extract the c lattice vector
        target_cell[1] = b_vector
    if which_vector.lower() == "c":
        c_vector = source_cell[2]  # Extract the c lattice vector
        target_cell[2] = c_vector

    # Set the modified cell back to the target_atoms
    target_atoms.set_cell(target_cell, scale_atoms=True)

    return target_atoms

############################################################################################


def make_transport_device(lead, scatter, bond_distance):

    lead.positions[:, 0] -= lead.positions[:, 0].min()
    scatter.positions[:, 0] -= scatter.positions[:, 0].min()

    scatter = set_lattice_vector(
        source_atoms=lead, target_atoms=scatter, which_vector="c")

    lead_cell = lead.get_cell()[:]
    scatter_cell = scatter.get_cell()[:]

    lead.positions[:, 0] -= lead.positions[:, 0].min()
    scatter.positions[:, 0] -= scatter.positions[:, 0].min()

    lead_width = lead.positions[:, 0].max() - lead.positions[:, 0].min()
    scatter_width = scatter.positions[:,
                                      0].max() - scatter.positions[:, 0].min()

    # bonds = None
    # lead_symbols = lead.get_chemical_symbols()
    # analysis = Analysis(lead)
    # if "C" in lead_symbols:
    #     bonds = analysis.get_bonds("C", "C", unique=True)
    #     bondsdists = analysis.get_values(bonds)
    # if "B" in lead_symbols and "N" in lead_symbols:
    #     bonds = analysis.get_bonds("B", "N", unique=True)
    #     bondsdists = analysis.get_values(bonds)
    # if "MO" in lead_symbols and "S" in lead_symbols:
    #     bonds = analysis.get_bonds("Mo", "S", unique=True)
    #     bondsdists = analysis.get_values(bonds)
    # bond_distance = min(min(bondsdists))
    # print("Bond Length =", bond_distance)

    left_lead, right_lead = lead.copy(), lead.copy()

    scatter.positions[:, 0] += (lead_width + bond_distance)

    combined = left_lead + scatter

    right_lead.positions[:, 0] += (lead_width +
                                   scatter_width + 2 * bond_distance)

    device = combined + right_lead

    device.set_cell([2*lead_cell[0]+scatter_cell[0],
                    lead_cell[1], lead_cell[2]])

    device.positions[range(len(lead)),
                     0] = left_lead.positions[range(len(lead)), 0]
    device.positions[range(len(lead)),
                     1] = left_lead.positions[range(len(lead)), 1]
    device.positions[range(len(lead)),
                     2] = left_lead.positions[range(len(lead)), 2]
    #
    device.positions[range(len(device)-len(lead), len(device)),
                     0] = right_lead.positions[range(len(lead)), 0]
    device.positions[range(len(device)-len(lead), len(device)),
                     1] = right_lead.positions[range(len(lead)), 1]
    device.positions[range(len(device)-len(lead), len(device)),
                     2] = right_lead.positions[range(len(lead)), 2]

    return device

############################################################################################


def help_func_01():
    pass

############################################################################################


def help_func_02():
    pass

############################################################################################


def help_func_03():
    pass

############################################################################################


def main():

    parser = argparse.ArgumentParser(
        description="Process different rotations of a molecule w.r.t a surface.")

    # parser.add_argument("--surface_poscar", type=str, default="CONTCAR_GRAPHENE_SURFACE", help="Surface (2D material) POSCAR or CONTCAR file")
    # parser.add_argument("--molecule_poscar", type=str, default="CONTCAR_BIOMOL_ALA", help="Molecule (e.g., amino acid) POSCAR or CONTCAR file")

    # parser.add_argument("--surface_poscar", type=str, default="CONTCAR_GRAPHENE_SUPERCELL", help="Surface (2D material) POSCAR or CONTCAR file")
    # parser.add_argument("--molecule_poscar", type=str, default="CONTCAR_BIOMOL_ALA", help="Molecule (e.g., amino acid) POSCAR or CONTCAR file")

    parser.add_argument("--surface_poscar", type=str, default="CONTCAR_GRAPHENE_NANOPORE",
                        help="Surface (2D material) POSCAR or CONTCAR file")
    parser.add_argument("--molecule_poscar", type=str, default="CONTCAR_BIOMOL_ALA",
                        help="Molecule (e.g., amino acid) POSCAR or CONTCAR file")
    parser.add_argument("--lead_poscar", type=str, default="POSCAR_GRAPHENE_RIBBON",
                        help="Electrode (e.g., nanoribbon) POSCAR or CONTCAR file")
    parser.add_argument("--inter_distance", type=float, default=2.5,
                        help="Vertical distance between surface and molecule (Ang)")
    parser.add_argument("--rotate_about_x", action="store_true",
                        default=True, help="Rotate about x-axis")
    parser.add_argument("--rotate_about_y", action="store_true",
                        default=True, help="Rotate about y-axis")
    parser.add_argument("--rotate_about_z", action="store_true",
                        default=True, help="Rotate about z-axis")
    parser.add_argument("--angle_increment", type=int,
                        default=30, help="Angle increment in units of degree")
    # parser.add_argument("--num_rotations", type=int, default=13, help="Number of rotation samplings")
    parser.add_argument("--view_structure", action="store_true",
                        default=False, help="Use ase gui to view structure")

    args = parser.parse_args()

    surface_poscar = args.surface_poscar
    molecule_poscar = args.molecule_poscar
    lead_poscar = args.lead_poscar
    inter_distance = args.inter_distance
    rotate_about_x = args.rotate_about_x
    rotate_about_y = args.rotate_about_y
    rotate_about_z = args.rotate_about_z
    angle_increment = args.angle_increment
    # num_rotations = args.num_rotations
    view_structure = args.view_structure

    # rotation about x-axis: theta angle: 0 - 360
    rotate_thetas = np.arange(0, 360, angle_increment)
    # rotation about y-axis: phi angle: 0 - 360
    rotate_phis = np.arange(0, 360, angle_increment)
    # rotation about z-axis: psi angle: 0 - 360
    rotate_psis = np.arange(0, 360, angle_increment)

    # rotate_thetas = np.linspace(0, 360, num_rotations, dtype=int)[:-1]
    # rotate_phis = np.linspace(0, 360, num_rotations, dtype=int)[:-1]
    # rotate_psis = np.linspace(0, 360, num_rotations, dtype=int)[:-1]

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
    lead_atoms = ase.io.read(f"{structures_dir}/{lead_poscar}")

    if view_structure:
        view(surface_atoms)
        view(molecule_atoms)
        view(lead_atoms)

    make_siesta_fdf(atoms=surface_atoms,
                    file_name=f"{surface_poscar}", run_task="sglpt", kpts_grid="1 1 1")

    make_siesta_fdf(atoms=molecule_atoms,
                    file_name=f"{molecule_poscar}", run_task="sglpt", kpts_grid="1 1 1")

    make_siesta_fdf(
        atoms=lead_atoms, file_name=f"{lead_poscar}", run_task="lead", kpts_grid="40 10 1")

    # lead_name = next((lead_poscar.replace(f"{tag}_", "") for tag in ["POSCAR", "CONTCAR", "LEAD", "ELEC"] if tag in lead_poscar.upper()), lead_poscar)
    # lead_atoms.write(f"{vaspposcars_dir}/POSCAR_LEAD_{lead_name}.vasp", format="vasp", vasp5=True, direct=True, sort=True)

    # for theta, phi, psi in rotate_angles:
    #
    #     combined_atoms = combine_surface_molecule(surface_atoms, molecule_atoms, inter_distance, theta=theta, phi=phi, psi=psi)
    #
    #     combined_name = make_combined_name(surface_poscar, molecule_poscar, theta=theta, phi=phi, psi=psi)
    #
    #     print("Combined Name: ", combined_name)
    #
    #     make_siesta_fdf(atoms=combined_atoms, file_name=f"POSCAR_{combined_name}", run_task="relax",  kpts_grid="1 1 1", make_constr=True, cutoff=8)
    #
    #     # combined_atoms.write(f"{vaspposcars_dir}/POSCAR_SCATTER_{combined_name}.vasp", format="vasp", vasp5=True, direct=True, sort=True)
    #     # combined_atoms.write(f"{snapshots_dir}/POSCAR_SCATTER_{combined_name}.png",  format="png")
    #
    #     make_siesta_fdf(atoms=combined_atoms, file_name=f"POSCAR_{combined_name}", run_task="sglpt",  kpts_grid="1 1 1")
    #
    #     make_siesta_fdf(atoms=combined_atoms, file_name=f"POSCAR_{combined_name}", run_task="scf",  kpts_grid="1 1 1")
    #
    #     device_atoms = make_transport_device(lead_atoms, combined_atoms, bond_distance=2.46855/2) # To do: calculate the bond_distance automatically?
    #
    #     make_siesta_fdf(atoms=device_atoms, file_name=f"POSCAR_{combined_name}", run_task="device",  kpts_grid="1 10 1")
    #     device_atoms.write(f"{vaspposcars_dir}/POSCAR_DEVICE_{combined_name}.vasp", format="vasp", vasp5=True, direct=True, sort=True)
    #
    # # Create GIF animation if the number of images is less than 100
    # if len(rotate_angles) < 100:
    #     output_gif_name = make_combined_name(surface_poscar, molecule_poscar)
    #     make_gif_with_pillow(image_folder=f"{snapshots_dir}", output_gif=f"{animations_dir}/{output_gif_name}.gif", duration=500)

############################################################################################


if __name__ == "__main__":
    main()
