# Credit Card Fraud Detection using Graph Convolutional Networks (GCN)

## Overview
This project demonstrates an approach to **Credit Card Fraud Detection** by modeling transaction data as a graph and utilizing a **Graph Convolutional Network (GCN)**. Traditional machine learning models can struggle with highly imbalanced datasets and the inherent relational complexities in financial transactions. GCNs offer a powerful way to leverage the graph structure of data (where transactions or users are nodes and connections represent relationships) to detect anomalies more effectively.

## Motivation & Objectives
The primary objectives of this project are:
* To **explore the application of Graph Neural Networks** (specifically GCNs) in financial fraud detection.
* To **build a robust classification model** capable of identifying fraudulent transactions amidst a vast majority of legitimate ones.
* To **address the challenges of imbalanced datasets** inherent in fraud detection using techniques like stratified sampling.
* To **showcase a comprehensive machine learning pipeline** from data loading and preprocessing to model training, evaluation, and performance analysis.

## Methodology & Approach
The solution pipeline involves several key steps:

1.  **Data Loading & Sampling:**
    * Loads the `creditcard_2023.csv` dataset, which contains anonymized transaction features (`V1` through `V28`), `Amount`, and `Class` (0 for non-fraudulent, 1 for fraudulent).
    * Performs **stratified sampling** to create a balanced subset of 10,000 transactions (5,000 fraudulent and 5,000 non-fraudulent) to mitigate imbalance issues during training.
2.  **Data Preprocessing:**
    * Applies **Standard Scaling** to numerical features to normalize their range, which is crucial for model convergence and performance.
3.  **KNN Graph Construction:**
    * Constructs a **K-Nearest Neighbors (KNN) graph** based on the cosine similarity of scaled transaction features. Each transaction is a node, and edges connect transactions that are similar to each other, forming a relational graph structure.
    * This graph structure is represented by an `edge_index` tensor, essential for GCNs.
4.  **PyTorch Geometric Data Object:**
    * Converts the preprocessed features (`X_scaled`), labels (`y`), and the constructed `edge_index` into a `torch_geometric.data.Data` object, which is the standard input format for PyTorch Geometric models.
5.  **Graph Neural Network (GNN) Model:**
    * A simple **Graph Convolutional Network (GCN)** model is defined with two `GCNConv` layers and ReLU activation, designed for node classification.
6.  **Training & Evaluation:**
    * The data is split into training and testing masks (80% training, 20% testing) using stratified sampling.
    * The GCN model is trained using the **Adam optimizer** and **CrossEntropyLoss** over 200 epochs.
    * Model performance is rigorously evaluated using:
        * **Accuracy**
        * **Classification Report** (Precision, Recall, F1-Score for each class)
        * **Confusion Matrix**
        * **ROC-AUC Score** (especially important for imbalanced classification tasks).

## Dataset
The dataset used is `creditcard_2023.csv`. It comprises transaction data where features `V1` through `V28` are the result of a PCA transformation due to confidentiality issues. The `Amount` feature is the transaction amount, and the `Class` feature indicates whether a transaction is fraudulent (1) or not (0).

## Model Architecture
The GCN model consists of:
* An input layer, taking the scaled transaction features.
* A first `GCNConv` layer, followed by a ReLU activation function.
* A second `GCNConv` layer, outputting 2 classes (fraud/non-fraud).

## Results
The model achieved high performance metrics on the test set, indicating its strong capability in distinguishing between fraudulent and legitimate transactions within the sampled dataset:
* **Test Accuracy:** ~0.9975
* **Precision (Fraud):** ~0.9980
* **Recall (Fraud):** ~0.9970
* **F1-Score (Fraud):** ~0.9975
* **ROC-AUC Score:** ~1.0000

These results highlight the effectiveness of GCNs in leveraging relational information for anomaly detection in financial datasets.

## How to Run
To replicate and run this analysis:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sreelakshmi-rv/](https://github.com/Sreelakshmi-rv/)[Your-New-Repo-Name].git
    ```
    (Replace `[Your-New-Repo-Name]` with the actual name you give to this repository, e.g., `Credit-Card-Fraud-Detection-GCN`.)
2.  **Navigate to the project directory:**
    ```bash
    cd [Your-New-Repo-Name]
    ```
3.  **Install necessary libraries:**
    * It's highly recommended to use a virtual environment.
    * ```bash
        python -m venv venv
        # On Windows:
        .\venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
        ```
    * Install dependencies:
        ```bash
        pip install pandas torch torch_geometric scikit-learn numpy matplotlib
        ```
4.  **Download the dataset:**
    * Obtain the `creditcard_2023.csv` dataset. (You can often find this on Kaggle or similar data platforms.)
    * Place the `creditcard_2023.csv` file in the same directory as the `Credit_Card_Fraud_Detection.ipynb` notebook.
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Credit_Card_Fraud_Detection.ipynb
    ```
    Follow the cells in the notebook to execute the full pipeline.

## Dependencies
* `pandas`
* `torch`
* `torch-geometric`
* `scikit-learn`
* `numpy`
* `matplotlib` (for plotting confusion matrix)

## Future Enhancements
* **More Complex GNN Architectures:** Explore advanced GNN models (e.g., GraphSAGE, GAT) or deeper architectures.
* **Dynamic Graphs:** Implement methods for handling dynamic or evolving transaction graphs.
* **Feature Engineering:** Integrate more domain-specific features or use autoencoders for anomaly detection.
* **Real-time Inference:** Optimize the model for faster predictions on new, incoming transactions.
* **Explainability:** Incorporate techniques to explain why certain transactions were flagged as fraudulent.
