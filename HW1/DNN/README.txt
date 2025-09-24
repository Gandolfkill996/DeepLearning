===========================================
Deep Neural Network Models (HW1 - DNN) Mac
===========================================

This folder contains implementations of several Deep Neural Network (DNN)
architectures for predicting "TARGET_deathRate". All models share the same
data preprocessing pipeline and follow a unified training/testing workflow.

-------------------------------------------
1. Training a DNN model
-------------------------------------------

Each model has its own Python script inside its subfolder, e.g.:

    HW1/DNN/DNN_30_16_8_4/DNN_30_16_8_4.py

To train a specific model, run from the project root:

    python3 -m HW1.DNN.DNN_30_16_8_4.DNN_30_16_8_4

This will:
- Load training data from: HW1/DNN/cancer_reg-1.csv
- Perform preprocessing (feature selection, scaling, NaN handling)
- Run grid search over hyperparameters:
    * dropout = {0.5, 0.7}
    * learning rate = {0.1, 0.01, 0.001, 0.0001}
    * batch size = {32, 64}
    * epochs = 50
- Select the best configuration based on **highest validation R²**
- Save the best model weights, hyperparameters, and feature list
- Save training/validation loss curves
- Save a results log of all tried configurations

Artifacts are saved in:

    HW1/DNN/<MODEL_NAME>/outputs/

For example, for DNN_30_16_8_4:

    HW1/DNN/DNN_30_16_8_4/outputs/

This folder will contain:
- best_<MODEL_NAME>_model_<timestamp>.pth    → trained PyTorch weights
- best_<MODEL_NAME>_params_<timestamp>.pkl   → best hyperparameters
- <MODEL_NAME>_features.pkl                  → selected feature list
- results_log.csv                            → performance log for all configs
- training_curve_<MODEL_NAME>_<timestamp>.png → train vs. validation MSE plot
- <MODEL_NAME>_predictions.csv               → predictions on test data

-------------------------------------------
2. Testing on new data (prediction only)
-------------------------------------------

To run a trained model on new data, use:

    python3 -m HW1.DNN.test_dnn <MODEL_NAME> <CSV_PATH>

Example:

    python3 -m HW1.DNN.test_dnn DNN_30_16_8_4 HW1/DNN/cancer_reg_new.csv

This will:
- Load the saved model and preprocessing artifacts from the model’s outputs/
- Apply the same feature selection and scaling as during training
- Generate predictions for all rows in the new dataset
- Save results into:

    HW1/DNN/<MODEL_NAME>/outputs/<MODEL_NAME>_predictions.csv

The output CSV will contain:
- First column: Predicted_TARGET_deathRate
- Remaining columns: all original columns from the new data

If the new dataset includes the true column "TARGET_deathRate",
the script will also compute and print MSE and R² against ground truth.

-------------------------------------------
3. Notes
-------------------------------------------

- All DNN models share the same preprocessing logic (Data_preprocess.py).
- Global best model is chosen by **highest validation R²**, not lowest MSE.
- Training includes early stopping with patience=10 epochs.
- Ensure new CSV files have consistent column names with training data.

-------------------------------------------
End of README
-------------------------------------------
