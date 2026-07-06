import pytest
import torch

from amulet.utils.__meta_classifiers import PermInvModel, meta_collate_fn


def test_permutation_invariance():
    # 2 layers: Layer1 (8 neurons, 4+1 dims), Layer2 (2 neurons, 8*H + 8+1 dims)
    # Actually PIM init calculates input_dim = dim + prev_N * H
    # Let's use simple dimensions.
    layer_shapes = [(8, 5), (2, 9)]
    model = PermInvModel(layer_shapes=layer_shapes, inside_dims=[16, 8])
    model.eval()

    # Create random weights for 1 sample
    # layer_params_list: List of [Batch=1, N_neurons, Dim]
    p1 = torch.randn(1, 8, 5)
    p2 = torch.randn(1, 2, 9)
    input_orig = [p1, p2]

    # Shuffle neurons in Layer 1
    perm = torch.randperm(8)
    p1_shuffled = p1[:, perm, :]
    input_shuffled = [p1_shuffled, p2]  # Only layer 1 is shuffled

    with torch.no_grad():
        out_orig = model(input_orig)
        out_shuffled = model(input_shuffled)

    assert torch.allclose(out_orig, out_shuffled, atol=1e-6)


def test_meta_collate_fn():
    # 2 models in batch. Each has 2 layers.
    m1_l1 = torch.randn(8, 4)
    m1_l2 = torch.randn(2, 8)
    m2_l1 = torch.randn(8, 4)
    m2_l2 = torch.randn(2, 8)

    batch = [([m1_l1, m1_l2], 0), ([m2_l1, m2_l2], 1)]

    batched_params, labels = meta_collate_fn(batch)

    assert labels.tolist() == [0, 1]
    assert len(batched_params) == 2
    # Layer 1: [Batch=2, N=8, D=4]
    assert batched_params[0].shape == (2, 8, 4)
    # Layer 2: [Batch=2, N=2, D=8]
    assert batched_params[1].shape == (2, 2, 8)
    # Check data content
    assert torch.equal(batched_params[0][0], m1_l1)
    assert torch.equal(batched_params[1][1], m2_l2)


def test_pim_batched_forward():
    layer_shapes = [(8, 4), (2, 8)]
    model = PermInvModel(layer_shapes=layer_shapes, inside_dims=[16, 4])

    # Batch of 3 models
    p1 = torch.randn(3, 8, 4)
    p2 = torch.randn(3, 2, 8)

    out = model([p1, p2])

    assert out.shape == (3, 1)


def test_pim_invalid_input_mismatch():
    model = PermInvModel(layer_shapes=[(8, 4)])
    # Input has 10 neurons instead of 8
    p1 = torch.randn(1, 10, 4)

    with pytest.raises(
        ValueError, match="expects 8 neurons"
    ):  # Specific ValueError from forward guard
        model([p1])


def test_pim_sum_pooling_exact_output():
    # Permutation invariance alone can't distinguish sum-pooling from mean- or
    # max-pooling (all three are order-independent). Pin the exact eval-mode
    # output so that swapping the documented sum-pooling for mean/max — which
    # changes every logit — fails here (mean pooling gives ~[-1.235, -2.518]).
    torch.manual_seed(0)
    model = PermInvModel(layer_shapes=[(3, 4)], inside_dims=[8, 4], n_classes=1)
    model.eval()
    params = [torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)]

    with torch.no_grad():
        out = model(params)

    torch.testing.assert_close(
        out.flatten(),
        torch.tensor([-2.772952, -6.621344]),
        atol=1e-5,
        rtol=0,
    )


def test_pim_conv2d_invariance():
    # Simulate a Conv2d layer [out=4, in=3, k=3, k=3] -> 4 "neurons", 27 features
    layer_shapes = [(4, 27)]
    model = PermInvModel(layer_shapes=layer_shapes)
    model.eval()

    p1 = torch.randn(1, 4, 27)

    # Permute the "neurons" (output channels)
    perm = torch.randperm(4)
    p1_shuffled = p1[:, perm, :]

    with torch.no_grad():
        out_orig = model([p1])
        out_shuffled = model([p1_shuffled])

    assert torch.allclose(out_orig, out_shuffled, atol=1e-6)
