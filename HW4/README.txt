Homework 4 – Graph Neural Networks (GCN)

This folder contains my implementation of GCN-1 layer and GCN-2 layer models for graph classification on the ENZYMES dataset. Both models were trained, evaluated, and compared following the assignment requirements (Step 1–6).

Project Structure
-----------------
HW4/
│
├── ENZYMES/                # Raw dataset files (A.txt, graph_indicator.txt, etc.)
│
├── GCN1.py                 # 1-layer GCN with Accuracy, F1, ROC-AUC
├── gcn1_model.pth          # Saved weights for GCN-1 model
├── gcn1_auc.png            # ROC-AUC plot for GCN-1 model
│
├── GCN2.py                 # 2-layer GCN with Accuracy, F1, ROC-AUC
├── gcn2_model.pth          # Saved weights for GCN-2 model
└── gcn2_auc.png            # ROC-AUC plot for GCN-2 model

File Descriptions
-----------------
1. GCN1.py
- Implements a 1-layer Graph Convolutional Network
- Includes:
  - GCN layer
  - Sum pooling
  - Cross-entropy training
  - Test evaluation with Accuracy, Macro F1 score, ROC-AUC
  - AUC curve saved as gcn1_auc.png

2. GCN2.py
- Implements a 2-layer GCN
- Includes:
  - Two GCN layers
  - Sum pooling
  - Test evaluation with Accuracy, Macro F1, ROC-AUC
- AUC curve saved as gcn2_auc.png

3. gcn1_model.pth / gcn2_model.pth
- Saved PyTorch model weights

4. gcn1_auc.png / gcn2_auc.png
- ROC-AUC curves generated during model testing

5. ENZYMES Dataset Folder
Contains raw dataset files:
ENZYMES_A.txt
ENZYMES_graph_indicator.txt
ENZYMES_graph_labels.txt
ENZYMES_node_attributes.txt
ENZYMES_node_labels.txt

How to Run
----------
Run the 1-layer GCN:
python GCN1.py

Run the 2-layer GCN:
python GCN2.py

Each script will:
1. Load the ENZYMES dataset
2. Train the model
3. Save model weights
4. Evaluate on the test set
5. Generate ROC-AUC plot

Evaluation Metrics
------------------
Both models report:
- Accuracy
- Macro F1 Score
- Macro ROC-AUC
- Saved ROC curves (PNG)