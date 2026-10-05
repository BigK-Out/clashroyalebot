import numpy as np

from train_plays import Clips


def test_clips_are_memory_mapped(tmp_path):
    X = np.random.randint(0, 255, (3, 8, 128, 128, 3), np.uint8)
    np.savez_compressed(tmp_path / "m1_note9.npz", X=X, y=np.array([1, 2, 122], np.int16), t=np.zeros(3, np.float32), proposed=np.ones(3, bool))
    ds = Clips([tmp_path / "m1_note9.npz"], augment=False)
    x, y, match, _ = ds[1]
    assert x.shape == (8, 3, 128, 128) and y == 2 and match == "m1_note9"  # one opponent per match+viewer
    assert isinstance(ds.data[tmp_path / "m1_note9.npz"][0], np.memmap)


def test_checkpoint_roundtrip_resumes_epoch_and_best(tmp_path):
    import torch

    from selfplay.model import PlayNet
    from train_plays import load_checkpoint, save_checkpoint

    net = PlayNet(5, pretrained=False)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=10)
    save_checkpoint(tmp_path / "last.pt", net, opt, sched, epoch=3, best=0.7)
    net2 = PlayNet(5, pretrained=False)
    opt2 = torch.optim.AdamW(net2.parameters(), 1e-3)
    sched2 = torch.optim.lr_scheduler.OneCycleLR(opt2, 1e-3, total_steps=10)
    start, best = load_checkpoint(tmp_path / "last.pt", net2, opt2, sched2)
    assert (start, best) == (4, 0.7)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), net2.state_dict().values()))


def test_unpacked_clips_are_not_stored_twice(tmp_path):
    X = np.random.randint(0, 255, (2, 8, 128, 128, 3), np.uint8)
    f = tmp_path / "m2_note14.npz"
    np.savez_compressed(f, X=X, y=np.array([3, 122], np.int16), t=np.zeros(2, np.float32), proposed=np.ones(2, bool))
    Clips([f], augment=False)
    assert "X" not in np.load(f).files  # pixels now live only in the .X.npy
    x, y, _, _ = Clips([f], augment=False)[0]  # and the dataset still loads from it
    assert y == 3 and np.allclose(x.numpy(), X[0].transpose(0, 3, 1, 2) / 255)


def test_resume_only_on_the_same_dataset(tmp_path):
    import torch

    from selfplay.model import PlayNet
    from train_plays import load_checkpoint, save_checkpoint

    net = PlayNet(5, pretrained=False)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=10)
    save_checkpoint(tmp_path / "last.pt", net, opt, sched, epoch=11, best=0.9, files=["a.npz", "b.npz"])
    # Same clips: continue. New clips (a later batch): start over instead of "already done".
    assert load_checkpoint(tmp_path / "last.pt", net, opt, sched, files=["a.npz", "b.npz"]) == (12, 0.9)
    assert load_checkpoint(tmp_path / "last.pt", net, opt, sched, files=["a.npz", "b.npz", "c.npz"]) == (0, 0.0)
