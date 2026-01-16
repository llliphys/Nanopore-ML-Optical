from pymatgen.core import Structure, Element, Composition


def get_chemical_features(poscar):
    # Load the POSCAR file
    struct = Structure.from_file(poscar)
    n_total = len(struct)

    # Initialize your feature dictionary
    features = {f"F{i}": 0 for i in [1, 2, 3, 8, 9, 11, 12, 13]}

    # Iterate through every atom in the structure
    for site in struct:
        el = site.specie  # This is a Pymatgen Element object

        features['F1'] += (el.valence[1] or 0)
        # features['F2'] += (el.atomic_mass or 0)
        # features['F3'] += (el.atomic_radius or 0)
        features['F8'] += (el.electron_affinity or 0)
        # features['F9'] += (el.vanderwaals_radius or 0)
        # features['F11'] += (el.average_ionic_radius or 0)
        features['F13'] += (el.ionization_energy or 0)

    # Calculate Averages
    averaged_features = {k: v / n_total for k, v in features.items()}
    averaged_features_clean = dict(
        filter(lambda item: abs(item[1]) > 1e-3, averaged_features.items()))

    return (averaged_features, averaged_features_clean)


# Example for Adenine Monophosphate
chemical_features, chemical_features_clean = get_chemical_features(
    "POSCAR_ALA")
print(chemical_features)

print(chemical_features_clean)

# def get_pymatgen_features(formula):
#     comp = Composition(formula)
#     n_total = comp.num_atoms

#     # Initialize feature accumulators
#     features = {f"F{i}": 0 for i in [1, 2, 3, 8, 9, 11, 12, 13]}

#     for element, amount in comp.items():
#         el = Element(element)

#         # Weighted accumulation
#         features['F1'] += (el.Z_valence or 0) * amount
#         features['F2'] += (el.atomic_mass or 0) * amount
#         features['F3'] += (el.X or 0) * amount
#         features['F8'] += (el.electron_affinity or 0) * amount
#         features['F9'] += (el.vanderwaals_radius or 0) * amount
#         features['F11'] += (el.average_ionic_radius or 0) * amount
#         features['F12'] += (el.atomic_radius or 0) * amount
#         features['F13'] += (el.ionization_energies[0]
#                             if el.ionization_energies else 0) * amount

#     # Divide by total atoms to get the "Average"
#     averaged_features = {k: v / n_total for k, v in features.items()}
#     return averaged_features


# # Example for Adenine Monophosphate
# adenine_features = get_pymatgen_features("C10H12N5O6P")
# print(adenine_features)
