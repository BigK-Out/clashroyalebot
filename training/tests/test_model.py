import torch

from selfplay.model import PlayNet


def test_shapes_for_full_and_early_clips():
    net = PlayNet(123, pretrained=False).eval()
    with torch.no_grad():
        assert net(torch.rand(2, 8, 3, 128, 128)).shape == (2, 123)
        assert net(torch.rand(2, 4, 3, 128, 128)).shape == (2, 123)


def test_learns_a_trivial_task():
    torch.manual_seed(0)
    net = PlayNet(3, pretrained=False)
    x = torch.zeros(6, 4, 3, 128, 128)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    for i in range(6):
        x[i, :, y[i]] = 1.0  # class k = channel k lit
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
    net.eval()
    assert (net(x).argmax(1) == y).all()
