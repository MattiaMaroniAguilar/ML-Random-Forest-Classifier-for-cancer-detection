# Cancer Prediction Model

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
