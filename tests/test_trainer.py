import pytest
from lexalign.finetuner.trainer import FinetuneTrainer, TrainerError

def test_trainer_initialization():
    config = {
        "model": {"path": "./models/test"},
        "dataset": {"path": "./data/test"},
        "training": {
            "method": "lora",
            "learning_rate": 3e-4,
            "num_epochs": 3
        },
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    assert trainer.config == config

def test_validate_model_path_exists(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    dataset_path = tmp_path / "data"
    dataset_path.mkdir()

    config = {
        "model": {"path": str(model_path)},
        "dataset": {"path": str(dataset_path)},
        "training": {"method": "lora"},
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    # Should not raise error
    trainer._validate_paths()

def test_validate_model_path_not_found():
    config = {
        "model": {"path": "/nonexistent/model"},
        "dataset": {"path": "./data/test"},
        "training": {"method": "lora"},
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    with pytest.raises(TrainerError, match="Model path not found"):
        trainer._validate_paths()

def test_get_output_dir_with_user_specified():
    config = {
        "model": {"path": "./models/test"},
        "dataset": {"path": "./data/test"},
        "training": {
            "method": "lora",
            "output_dir": "./custom/output"
        },
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    output_dir = trainer._get_output_dir()

    assert output_dir == "./custom/output"

def test_get_output_dir_default_timestamp(mocker):
    mocker.patch('lexalign.finetuner.trainer.datetime')

    config = {
        "model": {"path": "./models/gpt2"},
        "dataset": {"path": "./data/test"},
        "training": {"method": "lora"},
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    output_dir = trainer._get_output_dir()

    assert output_dir.startswith("./checkpoints/gpt2-")
