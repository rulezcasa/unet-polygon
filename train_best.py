import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConditionalUNet
from dataset import PolygonDataset
from torchvision import transforms
from tqdm import tqdm
import os
import wandb

# Global constants
color_vocab = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "cyan", "magenta"]
color_to_id = {c: i for i, c in enumerate(color_vocab)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loader setup
def get_dataloaders(batch_size=8):
    train_set = PolygonDataset("dataset/training", color_to_id)
    val_set = PolygonDataset("dataset/validation", color_to_id)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader

# Final training function
def train_best_model():
    # Best config
    lr = 0.001
    loss_name = "MSELoss"
    embed_dim = 16
    bilinear = True
    run_name = "Final_R23_best_lr0.001_lossMSELoss_dim16_bilinearTrue"

    # Initialize wandb
    wandb.init(
        project="polygon-unet",
        name=run_name,
        config={
            "learning_rate": lr,
            "loss_function": loss_name,
            "embed_dim": embed_dim,
            "bilinear": bilinear,
            "epochs": 50,
            "batch_size": 8
        }
    )

    print(f"\n[FINAL] Training best model config: {run_name}")
    train_loader, val_loader = get_dataloaders(batch_size=8)

    model = ConditionalUNet(color_vocab_size=len(color_vocab), embed_dim=embed_dim).to(device)
    model.up1.bilinear = bilinear
    model.up2.bilinear = bilinear
    model.up3.bilinear = bilinear

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(50):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/50] Training", leave=False)

        for x, y, color_id in train_bar:
            x, y, color_id = x.to(device), y.to(device), color_id.to(device)
            out = model(x, color_id)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss_avg = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"[Epoch {epoch+1}/50] Validation", leave=False)

        with torch.no_grad():
            for x, y, color_id in val_bar:
                x, y, color_id = x.to(device), y.to(device), color_id.to(device)
                out = model(x, color_id)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        val_loss_avg = val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg
        })

        print(f"[FINAL] Epoch {epoch+1:02d} - Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")

    # Save final model
    model_path = f"best_model/{run_name}.pth"
    os.makedirs("best_model", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    wandb.finish()
    print(f"\n✅ Final model saved to {model_path}")

# Run the final training
if __name__ == "__main__":
    train_best_model()
