import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from dataload.loader import build_feature_target_arrays, load_nanopore_dataframe


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        photon_energy = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)

        self.data_frame = pd.DataFrame(
            {
                "Theta": [10.0, 20.0],
                "Phi": [30.0, 40.0],
                "Psi": [50.0, 60.0],
                "Distance_Features": [
                    np.array([1.0, 2.0, 3.0], dtype=float),
                    np.array([4.0, 5.0, 6.0], dtype=float),
                ],
                "Transition_Energies": [
                    np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
                    np.array([0.5, 0.6, 0.7, 0.8], dtype=float),
                ],
                "Photon_Energy": [photon_energy, photon_energy],
                "Absorption_Coefficient": [
                    np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float),
                    np.array([15.0, 25.0, 35.0, 45.0, 55.0], dtype=float),
                ],
                "Amino_Acid": ["ALA", "GLY"],
            }
        )

    def test_load_nanopore_dataframe_reads_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_name = "toy_dataset"
            dataset_path = os.path.join(tmpdir, f"{dataset_name}.pkl")
            self.data_frame.to_pickle(dataset_path)

            loaded_df, loaded_name = load_nanopore_dataframe(tmpdir, dataset_name)

            self.assertEqual(loaded_name, dataset_name)
            self.assertEqual(len(loaded_df), len(self.data_frame))
            self.assertListEqual(list(loaded_df.columns), list(self.data_frame.columns))

    def test_build_feature_target_arrays_shapes_and_filtering(self) -> None:
        arrays = build_feature_target_arrays(
            df=self.data_frame,
            num_angle_feats=3,
            num_dist_feats=2,
            num_elec_feats=3,
            min_photo_energy=0.5,
            max_photo_energy=1.5,
        )

        self.assertEqual(arrays["X"].shape, (2, 8))
        self.assertEqual(arrays["Y"].shape, (2, 3))
        self.assertEqual(arrays["W"].shape, (2, 3))
        self.assertEqual(arrays["n_samples"], 2)
        self.assertEqual(arrays["n_features"], 8)
        self.assertEqual(arrays["n_targets"], 3)
        self.assertListEqual(
            arrays["feature_columns"],
            ["A1", "A2", "A3", "D1", "D2", "E1", "E2", "E3"],
        )
        np.testing.assert_allclose(arrays["W"][0], np.array([0.5, 1.0, 1.5]))
        np.testing.assert_allclose(arrays["Y"][0], np.array([20.0, 30.0, 40.0]))


if __name__ == "__main__":
    unittest.main()
