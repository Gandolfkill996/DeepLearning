import os
import sys
from .Linear_regression import test_model


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 -m HW1.Linear.test_linear <path_to_new_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        sys.exit(1)

    print(f"\n✅ Running test_model (Linear Regression) on {csv_path}")
    test_model(csv_path)


if __name__ == "__main__":
    main()