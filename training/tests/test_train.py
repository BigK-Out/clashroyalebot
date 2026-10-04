import numpy as np

from train_plays import Clips


def test_clips_are_memory_mapped(tmp_path):
    X = np.random.randint(0, 255, (3, 8, 128, 128, 3), np.uint8)
    np.savez_compressed(tmp_path / "m1_note9.npz", X=X, y=np.array([1, 2, 122], np.int16), t=np.zeros(3, np.float32), proposed=np.ones(3, bool))
    ds = Clips([tmp_path / "m1_note9.npz"], augment=False)
    x, y, match, _ = ds[1]
    assert x.shape == (8, 3, 128, 128) and y == 2 and match == "m1"
    assert isinstance(ds.data[tmp_path / "m1_note9.npz"][0], np.memmap)
