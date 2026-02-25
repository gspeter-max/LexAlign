import pytest
from lexalign.utils.device import DeviceManager, DeviceError

def test_detect_cuda_when_available(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    device, fell_back = manager.detect_device()

    assert device == "cuda"
    assert fell_back is False

def test_detect_cpu_when_cuda_unavailable(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = False

    manager = DeviceManager()
    device, fell_back = manager.detect_device()

    assert device == "cpu"
    assert fell_back is False

def test_override_device_with_cpu(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    device, fell_back = manager.get_device("cpu")

    assert device == "cpu"
    assert fell_back is False

def test_invalid_device_name(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')

    manager = DeviceManager()
    with pytest.raises(DeviceError, match="Invalid device"):
        manager.get_device("invalid")

def test_cuda_specification_when_unavailable(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = False

    manager = DeviceManager()
    device, fell_back = manager.get_device("cuda")

    assert device == "cpu"  # Fallback to CPU
    assert fell_back is True  # Warning flag set

def test_cuda_available_no_fallback(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    device, fell_back = manager.get_device("cuda")

    assert device == "cuda"
    assert fell_back is False


def test_get_device_none_returns_flat_tuple_not_nested(mocker):
    """Regression: get_device(None) was returning ((str, bool), False) — a nested tuple.

    This happened because detect_device() already returns a (str, bool) tuple,
    and get_device() was wrapping it again with ', False'.
    """
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    result = manager.get_device(None)

    # Must unpack as a flat two-tuple — not a nested structure
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple: {result}"
    device, fell_back = result
    assert device == "cuda"
    assert fell_back is False
    assert isinstance(device, str), f"device must be str, got {type(device)}"
    assert isinstance(fell_back, bool), f"fell_back must be bool, got {type(fell_back)}"


def test_get_device_none_cpu_returns_flat_tuple(mocker):
    """Regression companion: same check with CUDA unavailable."""
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = False

    manager = DeviceManager()
    device, fell_back = manager.get_device(None)

    assert device == "cpu"
    assert fell_back is False
