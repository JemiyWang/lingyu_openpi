import numpy as np
import pytest
import torch

from openpi.sed_robot_adapter import SedTeleAvatarInputs
from openpi.sed_robot_data_loader import _CachedSedLeRobotDataset
from openpi.training.data_loader import TransformedDataset


@pytest.mark.parametrize("shape", [(3, 8, 10), (1, 3, 8, 10), (1, 1, 3, 8, 10), (8, 10, 3)])
def test_image_layout(shape):
    image = torch.full(shape, 0.5)
    result = SedTeleAvatarInputs(action_dim=32)(
        {"images": {key: image for key in SedTeleAvatarInputs.required_rename_map}, "state": np.zeros(16)}
    )
    for value in result["image"].values():
        assert value.shape == (8, 10, 3)
        assert value.dtype == np.uint8
        assert np.all(value == 127)


def test_parallel_batch_preserves_order_and_transforms(monkeypatch):
    monkeypatch.setenv("OPENPI_SED_DECODE_THREADS", "4")

    class Dataset(_CachedSedLeRobotDataset):
        def __init__(self):
            pass

        def __getitem__(self, index):
            return {"value": index}

    dataset = Dataset()
    transformed = TransformedDataset(dataset, [lambda item: {"value": item["value"] * 2}])
    try:
        assert transformed.__getitems__([5, 2, 9, 1]) == [
            {"value": 10}, {"value": 4}, {"value": 18}, {"value": 2}
        ]
    finally:
        dataset._decode_pool.shutdown()


def test_regular_batch_fallback():
    transformed = TransformedDataset([{"value": 2}, {"value": 4}], [lambda item: item])
    assert transformed.__getitems__([1, 0]) == [{"value": 4}, {"value": 2}]
