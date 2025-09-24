===========================================
Linear Regression Model (HW1 - Linear) Mac
===========================================

This folder contains the implementation of a Linear Regression baseline
for predicting "TARGET_deathRate". The structure and workflow are aligned
with the DNN models for consistency.

-------------------------------------------
1. Training the Linear Regression model
-------------------------------------------

Run the following command from the project root (CS6073_DeepLearning):

    python3 HW1/Linear/Linear_regression.py

This will:
- Load training data from: HW1/Linear/cancer_reg-1.csv
- Perform preprocessing (feature selection, scaling, NaN handling)
- Train a Linear Regression model on train+val split
- Evaluate on the test set and print results
- Save all artifacts into: HW1/Linear/outputs/

Artifacts saved in outputs/:
- linear_model.pkl        → trained Linear Regression model
- scaler.pkl              → StandardScaler used for feature normalization
- selected_features.pkl   → list of selected feature names
- results_log.csv         → performance metrics (MSE, R², Adjusted R²)
- predictions.csv         → true vs predicted values on test set
- training_curve_linear.png → scatter plot of True vs Predicted (test set)

-------------------------------------------
2. Testing on new data (prediction only)
-------------------------------------------

To run the trained Linear Regression model on new data:

    python3 -m HW1.Linear.test_linear <CSV_PATH>

Example:

    python3 -m HW1.Linear.test_linear HW1/Linear/cancer_reg_new.csv

This will:
- Load the saved model and preprocessing artifacts from outputs/
- Apply the same feature selection and scaling as during training
- Generate predictions for all rows in the new dataset
- Save results into: HW1/Linear/outputs/linear_predictions.csv

The output CSV will contain:
- First column: Predicted_TARGET_deathRate
- Remaining columns: all original columns from the new data

-------------------------------------------
3. Notes
-------------------------------------------

- If the new dataset contains the column "TARGET_deathRate",
  the script will also compute and print MSE and R² for evaluation.
- If the column is missing, only predictions are generated.
- Ensure new data CSV has the same column names as the training data,
  otherwise some features may be skipped.
- All results and predictions are stored in the "outputs/" folder
  inside HW1/Linear.

-------------------------------------------
End of README
-------------------------------------------
