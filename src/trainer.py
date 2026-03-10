import copy
import os
from tqdm.auto import tqdm 
import torch


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience=10, save_path="checkpoints/best_model.pth"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.save_path = save_path
        
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
    

    def train(self, epochs):
        history = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  []}

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs",
                        bar_format="{l_bar}{bar:30}{r_bar}",
                        leave=True)

        for epoch in epoch_bar:
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()
            
            if self.scheduler:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.save_path)
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}",
                refresh=True
            )
            
            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
        self.model.load_state_dict(best_model_wts)
        return self.model, history
    

    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc="Train", leave=True, 
                    bar_format="{l_bar}{bar:20}{r_bar}")

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(
                loss=f"{running_loss/total:.4f}",
                acc=f"{correct/total:.4f}",
                refresh=True
            )

        return running_loss / total, correct / total


    def evaluate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.val_loader, desc="Val", leave=True, 
                    bar_format="{l_bar}{bar:20}{r_bar}")

        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)

                pbar.set_postfix(
                    loss=f"{running_loss/total:.4f}",
                    acc=f"{correct/total:.4f}",
                    refresh=True                       
                )

        return running_loss / total, correct / total