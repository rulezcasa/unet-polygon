import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConditionalUNet
from dataset import PolygonDataset
from torchvision import transforms
from tqdm import tqdm
import itertools
import multiprocessing
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

# Train a single config
def train_single_config(params):
    lr, loss_name, embed_dim, bilinear, run_id = params
    run_name = f"Run_{run_id}_lr{lr}_loss{loss_name}_dim{embed_dim}_bilinear{bilinear}"

    # Initialize wandb run
    wandb.init(
        project="polygon-unet",
        name=run_name,
        config={
            "learning_rate": lr,
            "loss_function": loss_name,
            "embed_dim": embed_dim,
            "bilinear": bilinear,
            "epochs": 30,
            "batch_size": 8,
            "run_id": run_id
        },
        reinit=True
    )

    print(f"\n[Run {run_id}] Starting with lr={lr}, loss={loss_name}, embed_dim={embed_dim}, bilinear={bilinear}")

    train_loader, val_loader = get_dataloaders(batch_size=8)

    model = ConditionalUNet(color_vocab_size=len(color_vocab), embed_dim=embed_dim).to(device)
    model.up1.bilinear = bilinear
    model.up2.bilinear = bilinear
    model.up3.bilinear = bilinear

    if loss_name == "L1Loss":
        criterion = nn.L1Loss()
    elif loss_name == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Unsupported loss")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    results = {}

    for epoch in range(15):  # 15 epochs for grid search
        model.train()
        total_loss = 0
        for x, y, color_id in train_loader:
            x, y, color_id = x.to(device), y.to(device), color_id.to(device)
            out = model(x, color_id)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss_avg = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, color_id in val_loader:
                x, y, color_id = x.to(device), y.to(device), color_id.to(device)
                out = model(x, color_id)
                loss = criterion(out, y)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg
        })

        print(f"[{run_name}] Epoch {epoch+1} - Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")

    # Save model
    model_path = f"models/{run_name}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)  # Log to wandb

    wandb.finish()

    return (run_id, lr, loss_name, embed_dim, bilinear, train_loss_avg, val_loss_avg)

# Grid values
lrs = [1e-4, 3e-4, 1e-3]
losses = ["L1Loss", "MSELoss"]
embed_dims = [8, 16]
bilinears = [True, False]

# Build parameter grid
grid = list(itertools.product(lrs, losses, embed_dims, bilinears))
grid_with_ids = [(lr, loss, dim, bilinear, f"R{str(i+1).zfill(2)}") for i, (lr, loss, dim, bilinear) in enumerate(grid)]

# Multiprocessing runner
if __name__ == "__main__":
    print(f"Running {len(grid_with_ids)} combinations with up to 4 workers.")
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(train_single_config, grid_with_ids)

    # Save summary
    with open("grid_search_summary.txt", "w") as f:
        for r in results:
            f.write(f"{r[0]} | lr={r[1]} | loss={r[2]} | embed_dim={r[3]} | bilinear={r[4]} | train_loss={r[5]:.4f} | val_loss={r[6]:.4f}\n")

    print("\n✅ Grid search complete. Summary saved to `grid_search_summary.txt`.")
