import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# ======================
# 1. Custom Dataset
# ======================
class RetinaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask


# ======================
# 2. Define U-Net
# ======================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        bridge = self.bridge(p4)

        u1 = self.up1(bridge)
        u1 = torch.cat([u1, c4], dim=1)
        c5 = self.conv1(u1)
        u2 = self.up2(c5)
        u2 = torch.cat([u2, c3], dim=1)
        c6 = self.conv2(u2)
        u3 = self.up3(c6)
        u3 = torch.cat([u3, c2], dim=1)
        c7 = self.conv3(u3)
        u4 = self.up4(c7)
        u4 = torch.cat([u4, c1], dim=1)
        c8 = self.conv4(u4)

        out = self.final(c8)
        return torch.sigmoid(out)


# ======================
# 3. Dice + IoU
# ======================
def dice_coef(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * inter + smooth) / (union + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


# ======================
# 4. Train / Test
# ======================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    train_dataset = RetinaDataset("Data/train/image", "Data/train/mask")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    best_dice = 0
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

        # simple validation (optional split or reuse test small subset)
        dice = dice_coef(pred, mask).item()
        print(f"Dice: {dice:.4f}")
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), "best_unet.pth")

    print("Training completed. Best Dice:", best_dice)


# ======================
# 5. Test Model
# ======================
def test_model(model_path="best_unet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = RetinaDataset("Data/test/image", "Data/test/mask")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dices, ious = [], []
    os.makedirs("predictions", exist_ok=True)

    with torch.no_grad():
        for i, (img, mask) in enumerate(test_loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            dice = dice_coef(pred, mask).item()
            iou = iou_score(pred, mask).item()
            dices.append(dice)
            ious.append(iou)

            # save prediction mask
            out_img = (pred[0,0].cpu().numpy() > 0.5).astype(np.uint8)*255
            Image.fromarray(out_img).save(f"predictions/pred_{i}.png")

    print(f"Mean Dice: {np.mean(dices):.4f}, Mean IoU: {np.mean(ious):.4f}")


if __name__ == "__main__":
    train_model()
    # After training, run:
    # test_model("best_unet.pth")
