

import sys
import pandas as pd
import joblib
import os

def predict_new_data(new_data_path):
    # Load model and pramters
    model = joblib.load("../HW1.Linear/linear_model.pkl")
    scaler = joblib.load("../HW1.Linear/scaler.pkl")
    selected_features = joblib.load("../HW1.Linear/selected_features.pkl")

    # Read new data
    df_new = pd.read_csv(new_data_path, encoding="latin1", encoding_errors="replace")

    # Deal with missing data, category data will be filled with most often appeared value,
    # numeric data will be filled with mean
    for col in df_new.columns:
        if df_new[col].dtype == "object":
            df_new[col] = df_new[col].fillna(df_new[col].mode()[0])
        else:
            df_new[col] = df_new[col].fillna(df_new[col].mean())

    # Get selected features
    X_new = df_new[selected_features]

    # use scaler to standardisation
    X_new_scaled = scaler.transform(X_new)

    # Predict
    y_pred = model.predict(X_new_scaled)

    # Get result
    df_new["Predicted_TARGET_deathRate"] = y_pred

    # Save output
    output_dir = os.path.dirname(new_data_path)
    output_path = os.path.join(output_dir, "../HW1.Linear/predictions.csv")
    df_new.to_csv(output_path, index=False)
    print(f"Prediction finished，result saved to {output_path}")
    print()
    print(df_new[["Predicted_TARGET_deathRate"]].head())
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Predict Method: python predict.py <newdata.csv>")
        sys.exit(1)

    new_data_path = sys.argv[1]
    predict_new_data(new_data_path)
