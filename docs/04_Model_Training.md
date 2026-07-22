# 04 - Model Training

---

# Overview

After building and compiling the neural network, the next step is to train the model.

Training is the process in which the neural network learns patterns from the training data by continuously adjusting its weights and biases.

The training process is performed using the `model.fit()` method.

```python
history = model.fit(
    train_images,
    train_labels,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
```

During training, TensorFlow repeatedly performs a sequence of operations known as the **training loop**.

---

# Why Do We Need Training?

Initially, the weights of a neural network are assigned random values.

Because of these random weights, the model produces random predictions.

Training gradually adjusts these weights so that the predictions become more accurate.

The objective of training is to minimize the prediction error (loss) while maximizing prediction accuracy.

---

# Training Workflow

```
Training Data

↓

Forward Pass

↓

Prediction

↓

Loss Calculation

↓

Backpropagation

↓

Gradient Calculation

↓

Weight Update

↓

Next Mini-Batch

↓

Repeat Until All Epochs Complete
```

---

# TensorFlow Implementation

```python
history = model.fit(
    train_images,
    train_labels,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint]
)
```

---

# Understanding model.fit()

The `fit()` method starts the complete training process.

It repeatedly

- Reads batches of training data
- Performs forward propagation
- Calculates the loss
- Performs backpropagation
- Updates trainable parameters
- Evaluates the model on the validation set
- Stores the training history

---

# Epoch

An **Epoch** means one complete pass through the entire training dataset.

Example

Training Samples

```
60000 Images
```

Batch Size

```
32
```

Number of batches

```
60000 / 32 ≈ 1875
```

One epoch consists of processing all **1,875 mini-batches**.

If

```python
epochs = 50
```

then the complete dataset is processed **50 times**.

---

# Batch Size

A Batch is a small subset of the training data processed at one time.

Instead of sending all 60,000 images together, the dataset is divided into smaller batches.

Example

```
Batch Size = 32

Batch 1

↓

32 Images

Batch 2

↓

32 Images

...

Batch 1875
```

Each batch produces one forward pass and one weight update.

---

# Iteration

An Iteration means processing **one mini-batch**.

Example

```
Batch Size = 32

60000 Images

↓

1875 Iterations

↓

1 Epoch
```

Relationship

```
Iterations × Batch Size

≈

Training Samples
```

---

# Forward Pass

During the forward pass, input images move through every layer of the network.

```
Images

↓

Flatten

↓

Dense

↓

Batch Normalization

↓

ReLU

↓

Dense

↓

Softmax

↓

Predicted Probabilities
```

No weights are updated during this step.

The model only makes predictions.

---

# Loss Calculation

The predicted probabilities are compared with the true labels.

Example

True Label

```
[0,0,0,1,0,0,0,0,0,0]
```

Prediction

```
[0.01,
0.03,
0.08,
0.71,
0.04,
...]
```

Categorical Crossentropy computes the prediction error.

A smaller loss indicates better predictions.

---

# Backpropagation

Once the loss is calculated, TensorFlow performs **Backpropagation**.

Backpropagation computes how much each trainable parameter contributed to the prediction error.

These gradients are then passed to the optimizer.

```
Loss

↓

Backpropagation

↓

Gradients

↓

Optimizer
```

---

# Weight Update

The optimizer updates

- Weights
- Biases
- BatchNorm Gamma
- BatchNorm Beta

using the gradients computed during backpropagation.

This happens after **every mini-batch**.

As training progresses

- Loss decreases
- Accuracy increases

---

# Validation

Training accuracy alone is not sufficient to judge a model.

Therefore, a portion of the training data is reserved for validation.

```python
validation_split=0.2
```

means

```
80%

↓

Training

20%

↓

Validation
```

The validation dataset is **not used to update the weights**.

It is used only to evaluate the model after each epoch.

---

# History Object

The `fit()` method returns a History object.

```python
history = model.fit(...)
```

It stores

```python
history.history.keys()
```

Typical output

```python
dict_keys([
'loss',
'accuracy',
'val_loss',
'val_accuracy'
])
```

These values are later used for visualization.

---

# Training Curves

Using the History object, we can plot

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

A good model generally shows

- Increasing Accuracy
- Decreasing Loss

---

# Early Stopping

Training for too many epochs may lead to overfitting.

EarlyStopping automatically stops training when the validation performance stops improving.

Example

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

### patience

Number of epochs to wait before stopping.

### restore_best_weights

Restores the weights from the epoch with the lowest validation loss.

---

# Model Checkpoint

ModelCheckpoint automatically saves the best model during training.

```python
ModelCheckpoint(
    "best_model.h5",
    save_best_only=True
)
```

Instead of saving every epoch, only the best-performing model is stored.

---

# Overfitting

Overfitting occurs when the model learns the training data too well but performs poorly on unseen data.

Typical signs

- Training accuracy continues increasing.
- Validation accuracy stops improving.
- Validation loss begins increasing.

Example

```
Epoch 20

Train Accuracy = 95%

Validation Accuracy = 92%

Epoch 45

Train Accuracy = 99%

Validation Accuracy = 90%
```

This indicates overfitting.

---

# Complete Training Pipeline

```
Build Model

↓

Compile Model

↓

Train Model

↓

Forward Pass

↓

Loss Calculation

↓

Backpropagation

↓

Gradient Calculation

↓

Weight Update

↓

Validation

↓

Next Epoch

↓

Training Complete
```

---

# Interview Questions

## What is model.fit()?

Starts the complete training process of the neural network.

---

## What is an Epoch?

One complete pass through the entire training dataset.

---

## What is a Batch?

A subset of the training data processed in one forward and backward pass.

---

## What is an Iteration?

Processing one mini-batch.

---

## When are weights updated?

After every mini-batch.

---

## Does validation data update the weights?

No.

Validation is used only to evaluate model performance.

---

## What is EarlyStopping?

A callback that stops training when validation performance no longer improves.

---

## Why use ModelCheckpoint?

To automatically save the best-performing model during training.

---

## What is the History object?

An object returned by `model.fit()` that stores training and validation metrics for every epoch.

---

## Summary

In this chapter, we learned how TensorFlow trains a neural network using `model.fit()`. We explored the concepts of epochs, batches, iterations, forward propagation, backpropagation, weight updates, validation, callbacks, and training history. Together, these components enable the model to learn meaningful patterns from the data while preventing overfitting and preserving the best-performing model.

---

# Key Takeaways

- `model.fit()` starts the training process.
- Training consists of repeated forward and backward passes.
- Weights are updated after every mini-batch.
- One epoch means processing the complete training dataset once.
- Validation data is never used for learning.
- The History object stores training metrics.
- EarlyStopping helps prevent overfitting.
- ModelCheckpoint saves the best model automatically.
