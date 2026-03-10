from dataclasses import dataclass

@dataclass
class TrainingConfig:

    data_dir: str = "/content/train"

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    batch_size: int = 32
    num_workers: int = 2
    min_resolution: int = 300
    
    num_classes: int = 12
    dropout: float = 0.2
    
    phase1_epochs: int = 50
    phase1_lr: float = 1e-3
    phase1_patience: int = 10
    
    phase2_epochs: int = 100
    phase2_lr: float = 1e-4
    phase2_patience: int = 15
    phase2_unfreeze_blocks: int = 2 
    
    seed: int = 42
    save_dir: str = "checkpoints"
