import pytest
from lexalign.utils.device import DeviceManager, DeviceError

def test_detect_cuda_when_available(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    device = manager.detect_device()

    assert device == "cuda"

def test_detect_cpu_when_cuda_unavailable(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = False

    manager = DeviceManager()
    device = manager.detect_device()

    assert device == "cpu"

def test_override_device_with_cpu(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    device = manager.get_device("cpu")

    assert device == "cpu"

def test_invalid_device_name(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')

    manager = DeviceManager()
    with pytest.raises(DeviceError, match="Invalid device"):
        manager.get_device("invalid")

def test_cuda_specification_when_unavailable(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = False

    manager = DeviceManager()
    device = manager.get_device("cuda")

    assert device == "cpu"  # Fallback to CPU
