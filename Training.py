import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from Model import MonoDepthModel
from DataLoader import NYUDataset
from Loss import *


def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5, save_path='monodepth_model.pth'):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (rgb_image, depth_map) in enumerate(train_loader):
            rgb_image, depth_map = rgb_image.to(device).float(), depth_map.to(device).float()

            optimizer.zero_grad()
            output_depth = model(rgb_image)
            loss = criterion(output_depth, depth_map)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_rgb_image, val_depth_map in val_loader:
                val_rgb_image, val_depth_map = val_rgb_image.to(device).float(), val_depth_map.to(device).float()
                val_output_depth = model(val_rgb_image)
                val_loss += criterion(val_output_depth, val_depth_map).item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((256, 320)),
    ])

    train_dataset = NYUDataset(split='train', transform=transform)
    val_dataset = NYUDataset(split='validation', transform=transform)

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MonoDepthModel().to(device)

    # criterion = nn.MSELoss()
    criterion = MonoDepthLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train(model, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10,
          save_path='monodepth_model.pth')
