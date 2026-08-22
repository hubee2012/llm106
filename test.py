import swanlab
import random

# Initialize a new swanlab experiment
swanlab.init(
  # Set the project for this experiment
  project="llm106_pretrain",
  workspace="llm106",
  # Track hyperparameters and experiment metadata
  config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10
  }
)

# Simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # Log training metrics to swanlab
  swanlab.log({"acc": acc, "loss": loss})

# [Optional] Finish the experiment. This is required in notebook environments.
swanlab.finish()
