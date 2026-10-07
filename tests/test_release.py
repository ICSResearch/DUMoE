import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative, name, imports=None):
    previous = sys.path[:]
    if imports is not None:
        sys.path.insert(0, str(ROOT / imports))
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


class ReleaseTests(unittest.TestCase):
    def test_activation_recomputation_preserves_weights_and_gradients(self):
        models = load_module("sci/sim/architecture/moe_duns.py", "memory_models")
        memory = load_module("sci/sim/training_memory.py", "memory_config")
        torch.manual_seed(0)
        model = models.DUMoE().train()
        recomputed = copy.deepcopy(model)
        keys = set(recomputed.state_dict())
        memory.enable_gradient_checkpointing(recomputed)
        self.assertEqual(keys, set(recomputed.state_dict()))
        measurement = torch.rand(1, 8, 62)
        mask = torch.rand(1, 28, 8, 62)
        outputs = []
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for network in (model, recomputed):
                prediction, auxiliary = network(measurement, mask)
                outputs.append(prediction.detach())
                (prediction.square().mean() + auxiliary).backward()
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for first, second in zip(model.parameters(), recomputed.parameters()):
                self.assertEqual(first.grad is None, second.grad is None)
                if first.grad is not None:
                    torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_all_entry_points_have_help_without_data(self):
        for script in (
            "ics/main_test.py",
            "csmri/main_test.py",
            "sci/sim/main_test.py",
            "sci/real/main_test.py",
            "sci/sim/train.py",
            "sci/real/train.py",
        ):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as folder:
                result = subprocess.run(
                    [sys.executable, str(ROOT / script), "--help"],
                    cwd=folder,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_measurement_dispersion_and_shape_validation(self):
        utils = load_module("sci/sim/utils.py", "sim_utils", "sci/sim")
        truth = torch.zeros(1, 28, 8, 8)
        truth[0, 3, 2, 1] = 0.5
        mask = torch.ones(1, 28, 8, 62)
        measured = utils.gen_meas_torch(truth, mask)
        self.assertEqual(tuple(measured.shape), (1, 8, 62))
        self.assertAlmostEqual(measured[0, 2, 7].item(), 1 / 28, places=7)
        self.assertEqual(torch.count_nonzero(measured).item(), 1)
        with self.assertRaises(ValueError):
            utils.gen_meas_torch(truth, mask[..., :-1])

    def test_sparse_router_rejects_unsupported_batches(self):
        models = load_module("sci/sim/architecture/moe_duns.py", "sim_models")
        model = models.DUMoE().eval()
        with self.assertRaisesRegex(ValueError, "one sample"):
            model(torch.zeros(2, 8, 62), torch.ones(2, 28, 8, 62))

    def test_simulation_training_loads_every_mat_file(self):
        training = load_module("sci/sim/training_data.py", "sim_training")
        with tempfile.TemporaryDirectory() as folder:
            for index in range(3):
                savemat(
                    Path(folder) / f"scene{index}.mat",
                    {"img": np.full((256, 256, 28), 32768, dtype=np.uint16)},
                )
            (Path(folder) / "README.md").write_text("ignored")
            images = training.load_training(folder)
            self.assertEqual(len(images), 3)
            self.assertTrue(np.all(images[0] == 0.5))
            patch = training.shuffle_crop(images, 1)
            self.assertEqual(tuple(patch.shape), (1, 28, 256, 256))
            self.assertTrue(torch.isfinite(patch).all())

    def test_real_training_crop_noise_is_finite(self):
        training = load_module("sci/real/training_data.py", "real_training")
        cube = np.full((96, 96, 28), 0.5, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.mat"
            savemat(path, {"mask_3d_shift": np.ones((96, 150, 28), dtype=np.float32)})
            dataset = training.TrainingDataset([cube], [cube], path, size=96, samples=2)
            measurement, label, mask = dataset[0]
            self.assertEqual(tuple(measurement.shape), (96, 150))
            self.assertEqual(tuple(label.shape), (28, 96, 96))
            self.assertEqual(tuple(mask.shape), (28, 96, 150))
            self.assertTrue(torch.isfinite(measurement).all())


if __name__ == "__main__":
    unittest.main()
