#!/bin/bash

#!/bin/bash

# Get current dict address
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# get into dict
cd "$PROJECT_DIR"

# activate venv
source venv/bin/activate

# check parameter
if [ $# -lt 1 ]; then
  echo "How to use: ./linear_predict.sh <new.csv>"
  exit 1
fi

# Predict
python Linear_predict.py "$1"

# If generate new predictions.csv，show first 5 rows
if [ -f predictions.csv ]; then
  echo "Prediction finished，outcome saved to predictions.csv"
  echo "Head of predictions look like："
  head -n 5 predictions.csv
fi

