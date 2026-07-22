# 03 - Model Compilation

---

# Overview

After designing the neural network architecture, the next step is to **compile the model**.

Building the model only defines the network structure. At this stage, TensorFlow knows:

- How many layers the model has
- The type of each layer
- The number of neurons
- The activation functions
- The trainable parameters

However, the model still does **not know**:

- How to measure prediction errors
- How to update the weights
- Which performance metrics to display

These details are provided using the `compile()` method.

---

# Why Do We Need Model Compilation?

A neural network cannot start training immediately after it is built.

Before training begins, TensorFlow needs answers to three important questions:

### 1. How should prediction errors be measured?

This is determined by the **Loss Function**.

---

### 2. How should the weights be updated?

This is determined by the **Optimizer**.

---

### 3. Which performance metrics should be displayed?

This is determined by the **Metrics**.

---

Only after answering these questions is the model ready for training.

---

# Model Compilation Workflow

```
Build Model

↓

Compile Model

↓

Train Model
```

---

# TensorFlow Implementation

```python
model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

Each parameter has a specific responsibility.

---

# Optimizer

```python
optimizer="sgd"
```

The optimizer decides **how the model updates its weights and biases** during training.

Without an optimizer, the model would calculate errors but would never learn from them.

---

# What is SGD?

SGD stands for

**Stochastic Gradient Descent**

It is one of the simplest and most widely used optimization algorithms.

Training process

```
Forward Pass

↓

Calculate Loss

↓

Backpropagation

↓

Compute Gradients

↓

Update Weights

↓

Next Mini-Batch
```

This process repeats for every mini-batch until all epochs are completed.

---

# Why is it called Stochastic?

Instead of updating the weights after processing the **entire dataset**, SGD updates the weights after processing **each mini-batch**.

Example

Dataset

```
60,000 Images
```

Batch Size

```
32
```

Number of batches

```
60000 / 32 ≈ 1875 batches
```

For every batch

```
Forward Pass

↓

Loss

↓

Backpropagation

↓

Weight Update
```

Therefore, the weights are updated approximately **1,875 times in one epoch**.

---

# Learning Rate

The optimizer updates the weights using a parameter called the **Learning Rate**.

The learning rate determines **how large each weight update should be**.

```
Small Learning Rate

↓

Slow Learning

↓

More Stable
```

```
Large Learning Rate

↓

Faster Learning

↓

May Overshoot the Minimum
```

Finding the right learning rate is one of the most important parts of training a neural network.

---

# Loss Function

```python
loss="categorical_crossentropy"
```

The loss function measures **how far the model's predictions are from the true labels**.

A smaller loss indicates better predictions.

```
Prediction

↓

Loss Function

↓

Error Value
```

---

# Why Categorical Crossentropy?

Our dataset has

```
10 classes
```

The labels are

```
One-Hot Encoded
```

Example

```
Sneaker

↓

[0,0,0,0,0,0,0,1,0,0]
```

The model predicts

```
[0.01,
0.02,
0.03,
0.01,
0.05,
0.02,
0.04,
0.79,
0.02,
0.01]
```

Categorical Crossentropy compares

- True probability distribution
- Predicted probability distribution

and calculates the prediction error.

The closer the prediction is to the true label, the smaller the loss.

---

# Metrics

```python
metrics=["accuracy"]
```

Metrics are used to monitor the model's performance during training.

Unlike the loss function, metrics do **not** update the model's weights.

They simply provide information about how well the model is performing.

Example

```
Epoch 1

Loss = 0.95

Accuracy = 71%
```

```
Epoch 10

Loss = 0.28

Accuracy = 91%
```

As training progresses

- Loss should decrease.
- Accuracy should increase.

---

# What Happens Internally During compile()?

The `compile()` method does **not** train the model.

Instead, TensorFlow prepares the model for training by configuring the required components.

Internally, TensorFlow:

- Registers the optimizer.
- Registers the loss function.
- Registers the evaluation metrics.
- Creates the training configuration.
- Builds the computation graph required for training.

No data is processed and no weights are updated during this step.

---

# Complete Workflow

```
Build Neural Network

↓

Compile Model

↓

Select Optimizer

↓

Select Loss Function

↓

Select Metrics

↓

Model Ready for Training

↓

model.fit()
```

---

# Interview Questions

## 1. Why do we use model.compile()?

It configures the model by specifying the optimizer, loss function, and evaluation metrics before training begins.

---

## 2. Does compile() train the model?

No.

It only prepares the model for training.

Training starts when `model.fit()` is called.

---

## 3. Why do we need an optimizer?

The optimizer updates the model's trainable parameters using the gradients computed during backpropagation.

Without an optimizer, the model cannot learn.

---

## 4. What is SGD?

SGD (Stochastic Gradient Descent) updates the model's parameters after processing each mini-batch.

---

## 5. Why is it called Stochastic?

Because parameter updates are performed using mini-batches rather than waiting for the entire dataset.

---

## 6. What is the purpose of the loss function?

The loss function measures how far the model's predictions are from the true labels.

---

## 7. Why do we use categorical_crossentropy?

Because Fashion MNIST is a multiclass classification problem with one-hot encoded labels.

---

## 8. What is the difference between Loss and Accuracy?

Loss is the value minimized during training.

Accuracy is a performance metric used to evaluate how many predictions are correct.

---

## 9. Does accuracy update the weights?

No.

Only the loss function and optimizer are involved in updating the model's weights.

---

## 10. What happens after compile()?

The model becomes ready for training using

```python
model.fit(...)
```

---

# Summary

In this chapter, we configured the neural network for training using the `compile()` method. We learned the purpose of the optimizer, loss function, and evaluation metrics, and explored how TensorFlow prepares the model before training begins.

Although `compile()` does not process any data or update the model's weights, it defines the complete training strategy that will be used during `model.fit()`.

---

# Key Takeaways

- Building a model defines the architecture.
- `compile()` defines the training strategy.
- The optimizer updates the model's weights.
- SGD performs weight updates after each mini-batch.
- The learning rate controls the size of each weight update.
- Categorical Crossentropy measures prediction error for multiclass classification.
- Accuracy is an evaluation metric, not a learning mechanism.
- Training begins only after calling `model.fit()`.
