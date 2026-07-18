# Machine Learning Random Forest Cancer Prediction Model
## Theoretical Foundation (What Makes Random Forest Powerful)

Ensemble Learning: Each model doesn't rely on a single decision tree, it creates a "forest" of 100 trees (based on your code) that each vote on the classification
Bagging (Bootstrap Aggregating): Each tree trains on a random subset of your data, preventing overfitting
Feature Randomness: Each tree considers only a random subset of features at each split, ensuring diversity
Majority Voting: The final prediction is what most trees agree on, this makes it robust against individual tree errors

## Model Information
- Type: Random Forest Classifier
- Task: Binary classification (Benign vs Malignant)
- Training Accuracy: 0.9649

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
- `README.md` - This file
