# Machine Learning Prediction: A Random Forest Classification Model Applied to Healthcare
![Alt text](/RFC_image.jpeg)
## The Clinical Challenge
Breast cancer is the most frequently diagnosed cancer in women globally. Because early intervention can improve survival rates to over 90%, developing rapid, highly accurate early detection systems is a critical medical priority. The ideal initial screening tool must be highly sensitive prioritizing a near zero false negative rate (maximizing recall) to ensure that no potential malignancies are missed during the first layer of detection.

## The Proposed Solution
To address this challenge, this project deploys an automated first line screening layer using a Random Forest classification algorithm.
Rather than relying on a single path of logic, Random Forest is an ensemble machine learning technique. It generates hundreds of distinct decision trees during training and aggregates their predictions to reach a final committee vote on the diagnosis.

## Why Random Forest?: Theoretical Foundation of Random Forest ML as proposed solution
This algorithm is particularly well suited for early cancer detection because it:

- Ensemble Learning: Each model doesn't rely on a single decision tree, it creates a "forest" of 100 trees (based on your code) that each vote on the classification
- Bagging (Bootstrap Aggregating): Each tree trains on a random subset of your data, preventing overfitting
- Feature Randomness: Each tree considers only a random subset of features at each split, ensuring diversity
- Majority Voting: The final prediction is what most trees agree on, this makes it robust against individual tree errors
- Pro/Cons: Highly accurate, but lacks traceability (trace exactly how one prediction was made)

## Dataset Context

- 569 total samples from breast biopsies
- 30 numerical features: measurements like radius, texture, perimeter, area, smoothness, etc.
- Binary classification: Benign (1) vs Malignant (0)
- 80/20 train-test split: 455 training, 114 test samples

## Model Information
- Type: Random Forest Classifier
- Task: Binary classification (Benign vs Malignant)
- Training Accuracy: 0.9649

## MLOps Workflow: Track > Train > Register > Deploy
- MLflow Experiment Tracking: Automatically logged all hyperparameters, metrics, and model artifacts.
- Model Registry and Hyperparameter Tuning: Registered the best model after testing different hyperparameters (50-100-200 trees config).
- Versioning: Models are versioned (v1) for reproducibility and maintainance.
- Model Portability: Created a standalone package that can be deployed anywhere.
- Evaluation Metrics: Accuracy, Feature Importance Analysis, Confusion Matrix

## Usage

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Load and use the model:
```python
import mlflow.pyfunc
import pandas as pd

# Load model
model = mlflow.pyfunc.load_model("./model")

# Make predictions
# Input: 30 features from breast cancer dataset
sample_data = pd.DataFrame(...)
prediction = model.predict(sample_data)
```

## Files
- `model/` - MLflow model artifacts
- `requirements.txt` - Python dependencies
