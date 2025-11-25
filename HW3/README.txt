HW3/
 ├── Data/
 │    ├── train/
 │    │    ├── image/
 │    │    └── mask/
 │    └── test/
 │         ├── image/
 │         └── mask/
 ├── screenshots/          # keep demo pictures
 ├── best_unet2_withBN.pth # best model
 ├── best_unet2_noBN.pth
 ├── best_unet3_withBN.pth
 ├── best_unet3_noBN.pth
 ├── U2N.py                # U-Net 2 blocks (with BatchNorm)
 ├── U2WN.py               # U-Net 2 blocks (without BatchNorm)
 ├── U3N.py                # U-Net 3 blocks (with BatchNorm)
 ├── U3WN.py               # U-Net 3 blocks (without BatchNorm)
 ├── generate_demo.py      # generate Step 6 model comparison demo
 ├── HW3-1.pdf             # HW
 └── README.txt

1. Dataset Description
-------------------------
- Task: Retina Blood Vessel Segmentation
- Input: RGB fundus image (~512×512 px)
- Label: Binary mask (1 = vessel, 0 = background)
- Split: 70 % train / 15 % validation / 15 % test (folders pre-split)

2. Pre-processing
--------------------
- Convert to RGB / grayscale
- Resize all images to 512×512 px
- Normalize pixel values to [0, 1]
- Convert masks to binary (threshold > 0.5 → 1)
- Optional: data augmentation (random flip / rotation)

3. Model Overview
--------------------
All scripts implement variants of U-Net:

| Script  | Blocks | Normalization | Saved Model |
|----------|---------|---------------|--------------|
| U2N.py   | 2 | ✅ BatchNorm | best_unet2_withBN.pth |
| U2WN.py  | 2 | ❌ No BatchNorm | best_unet2_noBN.pth |
| U3N.py   | 3 | ✅ BatchNorm | best_unet3_withBN.pth |
| U3WN.py  | 3 | ❌ No BatchNorm | best_unet3_noBN.pth |

Each network follows the encoder–decoder structure with skip connections.
Activation = ReLU, output = Sigmoid (binary segmentation).

4. Objective (Loss Function)
-------------------------------
Binary Cross-Entropy (BCE) is used as the training loss:
BCE = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
It penalizes pixel-wise classification errors for both vessel and background pixels.

5. Optimization
-----------------
Optimizer: Adam (lr = 1e-3)
Adam was selected for its adaptive learning rate and fast, stable convergence in deep networks.

6. Training
-----------------
Run any model script (e.g., U2N):
> python U2N.py

The script will:
1. Load training data from Data/train/
2. Train for 30 epochs (batch size = 2)
3. Save best model weights → best_unet2_withBN.pth
Training progress and Dice scores print to console (take screenshots for Step 6).

7. Testing
-------------
After training:
> python -c "from U2N import test_model; test_model('best_unet2_withBN.pth')"

Outputs per-image and mean Dice / IoU scores for the test set.

8. Evaluation Metrics
------------------------
- Dice coefficient: overlap between prediction & ground truth
- IoU (Intersection-over-Union): stricter overlap metric
Both range 0 – 1 (higher = better).

9. Model Demo (Step 6)
--------------------------
To generate comparison images across models:
> python generate_demo.py

This script loads the best weights of U-Net 2/3/4 blocks and produces figures:
screenshots/demo_0.png, demo_1.png, ...
Each figure shows Input | Label | U-Net 2 | U-Net 3 | U-Net 4, ready to insert into the report.

10. Notes
-------------
- All scripts are self-contained; only PyTorch + PIL + NumPy + tqdm required.
- GPU is recommended for faster training.
- Results and screenshots should be included in your submission ZIP as per HW3 instructions.
