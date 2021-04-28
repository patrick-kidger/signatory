import torch

from helpers import validation as v


tests = ['invert_signature']
depends = ['signature']
signatory = v.validate_tests(tests, depends)


def test_inverted_path_shape():
    """Tests that the inverted path is of the right shape"""
    for batch_size in (1, 2, 5):
        for input_stream in (2, 3, 10):
            for input_channels in (1, 2, 6):
                path = torch.rand((batch_size, input_stream, input_channels))
                for depth in (2, 4, 6):
                    signature = signatory.signature(path, depth)
                    inverted_path = signatory.invert_signature(signature, depth, input_channels, initial_position=path[:, 0, :])
                    assert inverted_path.shape == (batch_size, depth + 1, input_channels)


def test_initial_position_zero():
    """Tests that the inverted path initial position is the right one."""
    batch_size = 10
    input_stream = 10
    input_channels = 3
    path = torch.rand((batch_size, input_stream, input_channels))
    for depth in (2, 4, 6):
        signature = signatory.signature(path, depth)
        inverted_path = signatory.invert_signature(signature, depth, input_channels)
        assert torch.equal(inverted_path[:, 0, :], torch.zeros(batch_size, input_channels))
        initial_position = torch.rand((batch_size, input_channels))
        inverted_path = signatory.invert_signature(signature, depth, input_channels, initial_position=initial_position)
        assert torch.equal(inverted_path[:, 0, :], initial_position)


def test_inversion_close_original_path():
    """Tests that the the inverted path of a half circle is close to the original one. To be able to compare the two
    tensors, we take a depth equal to input_stream - 1 to have a reconstructed path that has exactly the same shape as
    the original one."""
    input_stream = 10
    time = torch.linspace(0, 1, input_stream)
    path = torch.stack([torch.cos(3.14 * time), torch.sin(3.14 * time)]).T.unsqueeze(0)
    input_channels = 2
    depth = input_stream - 1
    signature = signatory.signature(path, depth)
    inverted_path = signatory.invert_signature(signature, depth, input_channels, initial_position=path[:, 0, :])

    assert torch.allclose(path[:, :, :], inverted_path[:, :, :], atol=1e-01)

