import torch
from torch.utils.data import random_split, DataLoader

from src.config import TrainingConfig
from src.custom_dataset import CustomDataSet
from src.data_augmentation import DataAugmentation, TransformSubset
from src.model_architecture import WeedVGG16
from src.auto_finetuner import AutoFinetuner

def main():
    config = TrainingConfig()

    full_dataset = CustomDataSet(
        data_dir_path=config.data_dir,
        transform=None,      
        min_resolution=config.min_resolution
    )

    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size

    training_data, eval_data = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    aug = DataAugmentation()
    training_data = TransformSubset(training_data, aug.train_transform(config.mean, config.std))
    eval_data = TransformSubset(eval_data, aug.eval_transform(config.mean, config.std))

    train_loader = DataLoader(training_data, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(eval_data, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers, pin_memory=True)

    print(f"Total samples (after ≥{config.min_resolution}px filter) : {total}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    if total > 0:
        print(f"Classes ({len(full_dataset.classes)}): {full_dataset.classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = WeedVGG16(num_classes=config.num_classes, dropout=config.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params : {frozen_params:,}")

    finetuner = AutoFinetuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    best_model, phase1_history, phase2_history = finetuner.run()
    print("Auto-Finetuning Complete.")

if __name__ == "__main__":
    main()