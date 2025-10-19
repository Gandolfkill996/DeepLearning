-------------------------------------------
1️⃣  TRAINING THE MODEL
-------------------------------------------

To train the ConvNet models at different learning rates (0.1, 0.01, 0.001),
run the following commands in Terminal:

    cd HW2/ConvNet
    python ConvNet.py

During training, the script will:
  • Train three ConvNet models with different learning rates.
  • Save the best model (based on validation accuracy) to:
        outputs/lr_<learning_rate>/best_convnet_lr<learning_rate>.pth
  • Generate plots for loss, accuracy, ROC curve, and feature visualizations.
  • Print best Accuracy, F1-score, and AUC for each model in Terminal.

-------------------------------------------
2️⃣  TESTING THE MODEL (using test_model)
-------------------------------------------

After training, you can evaluate the best ConvNet model on **10% of the MNIST test dataset**
using the `test_model()` function.

Example (for the model trained with lr = 0.1):

    cd HW2/ConvNet
    python ConvNet.py

Then in Terminal (interactive Python mode or inside the main block):

    >>> from ConvNet import test_model
    >>> import torch
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    >>> test_model("outputs/lr_0.001/best_convnet_lr0.001.pth", device)

Or directly run the test block included in ConvNet.py:
    python ConvNet.py --test

This function will:
  • Load the trained model weights from the specified .pth file.
  • Load the MNIST test dataset (10% subset only).
  • Evaluate Accuracy, F1-score, and AUC.
  • Generate and save a new ROC curve at:
        outputs/test_eval/roc_curve.png
  • Print metrics to Terminal, e.g.:
        ✅ Test Results -> Accuracy: 0.9734, F1: 0.9729, AUC: 0.9986

-------------------------------------------
3️⃣  OUTPUT FILES
-------------------------------------------

Each learning rate folder (lr_0.1, lr_0.01, lr_0.001) contains:
  - best_convnet_lr*.pth        → Saved best model weights.
  - accuracy_curve.png          → Train/validation accuracy curve.
  - loss_curve.png              → Train/validation loss curve.
  - roc_curve.png               → ROC curve of validation/test dataset.
  - layer1_features.png         → Visualization of conv layer 1 filters.
  - layer2_features.png         → Visualization of conv layer 2 activations.

-------------------------------------------
4️⃣  DEVICE SUPPORT
-------------------------------------------
The code automatically detects device:
  • NVIDIA GPU → CUDA
  • Apple Silicon → MPS
  • Otherwise → CPU

You will see:
    ✅ Using device: NVIDIA GeForce RTX 3060
or
    ✅ Using device: Apple MPS GPU

-------------------------------------------
5️⃣  NOTES
-------------------------------------------
• MNIST dataset will be automatically downloaded to:
    ../data/MNIST/
• All training and testing outputs are saved in `outputs/`.
• `.gitignore` excludes data, cache, and output folders from GitHub uploads.
• You can modify batch size, epochs, or learning rates inside ConvNet.py for experiments.