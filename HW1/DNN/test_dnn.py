import sys
from HW1.DNN.DNN_base import test_model

# Import all models here
from HW1.DNN.DNN_16.DNN_16 import DNN_16
from HW1.DNN.DNN_30_8.DNN_30_8 import DNN_30_8
from HW1.DNN.DNN_30_16_8.DNN_30_16_8 import DNN_30_16_8
from HW1.DNN.DNN_30_16_8_4.DNN_30_16_8_4 import DNN_30_16_8_4

# Map model names to classes
MODEL_MAP = {
    "DNN_16": DNN_16,
    "DNN_30_8": DNN_30_8,
    "DNN_30_16_8": DNN_30_16_8,
    "DNN_30_16_8_4": DNN_30_16_8_4,
}


def main():
    """
    Usage:
        python3 -m HW1.DNN.test_dnn <MODEL_NAME> <CSV_PATH>

    Example:
        python3 -m HW1.DNN.test_dnn DNN_30_16_8_4 HW1/DNN/cancer_reg_new.csv
    """
    if len(sys.argv) != 3:
        print("   Usage: python3 -m HW1.DNN.test_dnn <MODEL_NAME> <CSV_PATH>")
        print("   Example: python3 -m HW1.DNN.test_dnn DNN_30_16_8_4 HW1/DNN/cancer_reg_new.csv")
        sys.exit(1)

    model_name, csv_path = sys.argv[1], sys.argv[2]

    if model_name not in MODEL_MAP:
        print(f" Unknown model name: {model_name}")
        print(f"Available models: {list(MODEL_MAP.keys())}")
        sys.exit(1)

    ModelClass = MODEL_MAP[model_name]

    print(f" Running test_model with {model_name} on {csv_path}")
    test_model(ModelClass, model_name, new_data_path=csv_path)


if __name__ == "__main__":
    main()
