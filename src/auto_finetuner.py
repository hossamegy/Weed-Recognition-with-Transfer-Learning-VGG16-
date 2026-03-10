import os
import torch
import torch.nn as nn
from src.trainer import Trainer
from src.config import TrainingConfig

class AutoFinetuner:
    def __init__(self, model, train_loader, val_loader, device, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        
    def run(self):
        print("=== Phase 1: Training Classifier Head ===")
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.phase1_lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        save_path = os.path.join(self.config.save_dir, "phase1_best.pth")
        trainer = Trainer(
            self.model, self.train_loader, self.val_loader, 
            self.criterion, optimizer, scheduler, self.device, 
            patience=self.config.phase1_patience, save_path=save_path
        )
        self.model, history1 = trainer.train(epochs=self.config.phase1_epochs)
        
        print("\n=== Phase 2: Progressive Unfreezing ===")
        params = list(self.model.features.parameters())
        num_unfreeze = self.config.phase2_unfreeze_blocks * 4
        
        if num_unfreeze > 0 and num_unfreeze <= len(params):
            unfreeze_params = params[-num_unfreeze:]
            for param in unfreeze_params:
                param.requires_grad = True
                
            optimizer = torch.optim.RMSprop(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.phase2_lr
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            
            save_path = os.path.join(self.config.save_dir, "phase2_best.pth")
            trainer = Trainer(
                self.model, self.train_loader, self.val_loader, 
                self.criterion, optimizer, scheduler, self.device, 
                patience=self.config.phase2_patience, save_path=save_path
            )
            self.model, history2 = trainer.train(epochs=self.config.phase2_epochs)
        else:
            print("Skipping Phase 2, unfreeze blocks is 0 or exceeds layers.")
            history2 = None
            
        return self.model, history1, history2
