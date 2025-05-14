# Concise Workflow
# Linear Regression with PyTorch

import torch
from torch import nn
import matplotlib.pyplot as plt

torch.__version__

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
X[:10], y[:10]

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

plot_predictions(X_train, y_train, X_test, y_test)

# Create linear module
class LinearRegressionModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    # Use nn.Linear()
    self.linear_layer = nn.Linear(in_features=1,
                                  out_features=1)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)

torch.manual_seed(42)
model1 = LinearRegressionModelV2()
model1, model1.state_dict()

# Training
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 200

for epoch in range(epochs):
  model1.train()
  # Forward pass
  y_pred = model1(X_train)

  # Loss Calculation
  loss = loss_fn(y_pred, y_train)

  # Optimizer zero grad
  optimizer.zero_grad()

  # Perform backpropogation
  loss.backward()

  # Optimiser Step
  optimizer.step()

  # Testing
  model1.eval()
  with torch.inference_mode():
    test_pred = model1(X_test)
    test_loss = loss_fn(test_pred, y_test)
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

model1.state_dict()

plot_predictions(X_train, y_train, X_test, y_test)

model1.eval()

with torch.inference_mode():
  y_preds = model1(X_test)
y_preds

plot_predictions(predictions=y_preds)

# Saving and Loading State Dict of Model
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "02PyTorch.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

MODEL_SAVE_PATH
torch.save(obj=model1.state_dict(), f=MODEL_SAVE_PATH)

model1.state_dict()

loaded_model = LinearRegressionModelV2()

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model

loaded_model.eval()
with torch.inference_mode():
  loaded_preds = loaded_model(X_test)
loaded_preds == y_preds

# If done correctly should return only True
