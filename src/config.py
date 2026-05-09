from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = Path(r"C:\Users\qasim\Desktop\PSL Stuff\Pakistan Sign Language Urdu Alphabets")
    split_dir: Path = project_root / "splits"
    checkpoints_dir: Path = project_root / "checkpoints"
    reports_dir: Path = project_root / "reports"


@dataclass(frozen=True)
class StaticTrainConfig:
    image_size: int = 299
    num_classes: int = 36
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class DynamicTrainConfig:
    image_size: int = 224
    num_classes: int = 4
    num_frames: int = 30
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True

    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 2
    dropout: float = 0.3
    attention_dim: int = 256


PATHS = Paths()
STATIC_CFG = StaticTrainConfig()
DYNAMIC_CFG = DynamicTrainConfig()
