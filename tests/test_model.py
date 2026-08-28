import torch

from src.model import get_model


def test_model_output_shape():
    model = get_model("cnn", 10)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)
