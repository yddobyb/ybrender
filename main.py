from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import joblib

# The model definition must match the training definition.
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)

# Enforces strict types for request and provides examples for FastAPI documentation. 
app = FastAPI(title="FastAPI + PyTorch Iris Predictor")

# Load scaler and model once at startup
scaler = joblib.load("iris_scaler.joblib")

model = IrisNet()
model.load_state_dict(torch.load("iris_model.pth", map_location="cpu"))
model.eval()

class_names = ["setosa", "versicolor", "virginica"]

class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs to try the predictor."}

@app.post("/predict")
def predict(req: IrisRequest):
    # Convert request to model input
    x = [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]
    x_scaled = scaler.transform(x)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)

    # Run model
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(torch.argmax(logits, dim=1).item())

    # Return JSON response
    return {
        "predicted_class": class_names[pred_idx],
        "class_index": pred_idx,
        "probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        },
    }
