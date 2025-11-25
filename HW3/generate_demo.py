import sys
import os
sys.path.append(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(__file__)
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, random

# ======== 导入四个模型结构 ========
from U2WN import UNet_2block_noBN as U2WN_Model
from U2N import UNet_2block as U2N_Model
from U3WN import UNet_3block_noBN as U3WN_Model
from U3N import UNet_3block as U3N_Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== define model's info table ========
model_info = [
    ("U2WN", U2WN_Model, os.path.join(BASE_DIR, "./best_unet2_noBN.pth")),
    ("U2N",  U2N_Model,  os.path.join(BASE_DIR, "./best_unet2_withBN.pth")),
    ("U3WN", U3WN_Model, os.path.join(BASE_DIR, "./best_unet3_noBN.pth")),
    ("U3N",  U3N_Model,  os.path.join(BASE_DIR, "./best_unet3_withBN.pth")),
]

# ======== Randomly select 4 test picture ========
test_images = sorted(os.listdir("Data/test/image"))
test_masks  = sorted(os.listdir("Data/test/mask"))
os.makedirs("screenshots", exist_ok=True)

sample_indices = random.sample(range(len(test_images)), 4)

# ======== generate models' predictions and save =======
for tag, ModelClass, weight_path in model_info:
    print(f"\n=== Generating demo for {tag} ===")
    model = ModelClass().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    for k, idx in enumerate(sample_indices):
        img_path  = os.path.join("Data/test/image", test_images[idx])
        mask_path = os.path.join("Data/test/mask",  test_masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)/255.,
                                  dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = (model(img_tensor)[0, 0].cpu() > 0.5).numpy()

        # ======== Draw Input / Label / Prediction ========
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Input"); plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Label"); plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title(f"{tag} Prediction"); plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"screenshots/demo_{tag}_{k}.png", dpi=200)
        plt.close()

print("\n✅ All demo images saved in screenshots/ folder.")

