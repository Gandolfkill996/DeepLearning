-------------------------------------------
1️⃣  TRAINING THE MODEL
-------------------------------------------

To train the DNN models at different learning rates (0.1, 0.01, 0.001),
run the following command in Terminal:

    cd HW2/DNN
    python DNN.py

During training, the script will:
  • Train models at 3 learning rates.
  • Save the best model (based on validation accuracy) to:
        outputs/lr_<learning_rate>/best_dnn_lr<learning_rate>.pth
  • Generate plots for loss, accuracy, ROC curve, and feature visualization.
  • Print best Accuracy, F1, and AUC for each model in Terminal.

-------------------------------------------
2️⃣  TESTING THE MODEL (using test_model)
-------------------------------------------

After training, you can evaluate the best model on **10% of the MNIST test dataset**
by running the built-in `test_model()` function.

Example (for the model trained with lr = 0.1):

    cd HW2/DNN
    python DNN.py

Then in Terminal (interactive mode or inside DNN.py main block):

    >>> from DNN import test_model
    >>> import torch
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    >>> test_model("outputs/lr_0.001/best_dnn_lr0.001.pth", device)

Or directly run the test block included in DNN.py:
    python DNN.py --test

This function will:
  • Load the saved model weights from the given path.
  • Load the MNIST test dataset (10% subset).
  • Evaluate accuracy, F1-score, and AUC.
  • Generate and save a new ROC curve at:
        outputs/test_eval/roc_curve.png
  • Print metrics to Terminal:
        ✅ Test Results -> Accuracy: 0.9654, F1: 0.9648, AUC: 0.9982

-------------------------------------------
3️⃣  OUTPUT FILES
-------------------------------------------

Each learning rate folder (lr_0.1, lr_0.01, lr_0.001) contains:
  - best_dnn_lr*.pth        → Saved best model weights.
  - accuracy_curve.png      → Train/validation accuracy curve.
  - loss_curve.png          → Train/validation loss curve.
  - roc_curve.png           → ROC curve of validation/test dataset.
  - layer1_features.png     → Visualization of layer 1 activations.
  - layer2_features.png     → Visualization of layer 2 activations.

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
• All training results are saved in `outputs/`.
• `.gitignore` excludes all data, cache, and outputs from GitHub uploads.