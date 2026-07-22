# 05 - Hyperparameter Tuning

---

# Overview

Building a neural network is only the first step. The performance of the model also depends on selecting appropriate **hyperparameters**.

Hyperparameters are values that are specified **before training begins** and control how the model learns. Choosing the right hyperparameters can significantly improve the model's accuracy and generalization.

Instead of manually trying different combinations, we use **Keras Tuner** to automatically search for the best hyperparameter values.

---

# What are Hyperparameters?

Hyperparameters are configuration settings that are chosen before training starts.

Examples include

- Number of hidden layers
- Number of neurons
- Activation functions
- Optimizer
- Learning rate
- Dropout rate
- Regularization method
- Batch size
- Number of epochs

Unlike weights and biases, hyperparameters are **not learned during training**.

---

# Why Do We Need Hyperparameter Tuning?

Different hyperparameter combinations produce different model performances.

For example

```
Model A

Hidden Layers = 2

Accuracy = 89%
```

```
Model B

Hidden Layers = 3

Accuracy = 92%
```

Even though both models use the same dataset, changing the hyperparameters improves the performance.

The objective of hyperparameter tuning is to automatically discover the best-performing combination.

---

# Hyperparameter Tuning Workflow

```
Define Search Space

↓

Generate Trial

↓

Build Model

↓

Compile Model

↓

Train Model

↓

Evaluate Validation Performance

↓

Store Results

↓

Generate Next Trial

↓

Repeat

↓

Select Best Hyperparameters
```

---

# Keras Tuner

TensorFlow provides **Keras Tuner**, a library that automatically searches for the best hyperparameters.

Example

```python
import keras_tuner as kt
```

---

# Bayesian Optimization

In this project, we use

```python
kt.BayesianOptimization()
```

Bayesian Optimization is an intelligent search algorithm.

Instead of trying random combinations, it learns from previous trials and predicts which hyperparameter combinations are more likely to improve performance.

Advantages

- Faster than Grid Search
- Smarter than Random Search
- Requires fewer experiments
- Finds good models efficiently

---

# Defining the Search Space

A search space defines all possible hyperparameter values.

Example

```python
hp.Int("n_layers", 1, 4)
```

Possible values

```
1

2

3

4
```

The tuner chooses one value during each trial.

---

## Integer Hyperparameters

```python
hp.Int(
    "units_0",
    min_value=16,
    max_value=256,
    step=16
)
```

Possible values

```
16

32

48

...

256
```

---

## Choice Hyperparameters

```python
hp.Choice(
    "activation_0",
    ["relu", "tanh"]
)
```

Possible values

```
relu

tanh
```

---

## Float Hyperparameters

```python
hp.Float(
    "dropout_0",
    0.0,
    0.5,
    step=0.1
)
```

Possible values

```
0.0

0.1

0.2

...

0.5
```

---

# Dynamic Hidden Layers

Instead of fixing the architecture, we allow the tuner to choose the number of hidden layers.

```python
n_layers = hp.Int(
    "n_layers",
    1,
    4
)
```

Then

```python
for i in range(n_layers):
```

The tuner dynamically creates

```
1 Layer

or

2 Layers

or

3 Layers

or

4 Layers
```

---

# Active vs Inactive Hyperparameters

This is one of the most commonly misunderstood concepts.

Suppose

```
Maximum Layers = 4
```

The search space contains

```
units_0

units_1

units_2

units_3
```

Now assume the tuner chooses

```
n_layers = 2
```

The model actually uses only

```
units_0

units_1
```

The remaining hyperparameters

```
units_2

units_3
```

still appear inside

```python
best_hps.values
```

but they are **inactive** because the model has only two hidden layers.

---

# Example

Suppose

```python
best_hps.values
```

returns

```python
{
'n_layers':2,

'units_0':128,

'units_1':64,

'units_2':96,

'units_3':160
}
```

Only

```
128

64
```

are used to build the final model.

The remaining values are ignored.

---

# Running the Search

```python
tuner.search(
    train_images,
    train_labels,
    validation_split=0.2,
    callbacks=[early_stop]
)
```

During tuning

- Multiple models are trained.
- Validation accuracy is recorded.
- Results are stored.
- The next trial is generated.

---

# Retrieving the Best Hyperparameters

```python
best_hps = tuner.get_best_hyperparameters(
    num_trials=1
)[0]
```

This returns the highest-performing hyperparameter combination discovered during the search.

---

# Building the Final Model

After selecting the best hyperparameters

```python
best_model = tuner.hypermodel.build(best_hps)
```

The final model is built using those values.

---

# Why Retrain the Best Model?

During tuning, each trial is trained only long enough to compare models.

After selecting the best hyperparameters, we train the final model again using the complete training process.

This produces the final model used for evaluation and prediction.

---

# Callbacks During Tuning

During hyperparameter tuning we used

```python
EarlyStopping()
```

Only.

Reason

The objective is simply to compare models efficiently.

Saving every trial would consume unnecessary storage.

---

# Callbacks During Final Training

After selecting the best hyperparameters

```python
callbacks=[
    EarlyStopping(),
    ModelCheckpoint()
]
```

Reason

Now we want to save the best version of the final model.

---

# Complete Hyperparameter Tuning Workflow

```
Define Search Space

↓

Bayesian Optimization

↓

Generate Trial

↓

Build Model

↓

Compile Model

↓

Train

↓

Validation Score

↓

Store Results

↓

Repeat

↓

Best Hyperparameters

↓

Build Final Model

↓

Train Final Model

↓

Save Best Model
```

---

# Interview Questions

## What is a hyperparameter?

A hyperparameter is a configuration value chosen before training begins.

---

## What is the difference between a parameter and a hyperparameter?

Parameters

- Learned during training
- Example: weights and biases

Hyperparameters

- Chosen before training
- Example: learning rate, number of layers

---

## Why use Keras Tuner?

To automatically search for the best hyperparameter combination.

---

## Why Bayesian Optimization?

Because it intelligently selects promising hyperparameter combinations instead of testing every possibility.

---

## What is a search space?

The collection of all possible hyperparameter values that the tuner can explore.

---

## Why do inactive hyperparameters appear in best_hps?

The tuner stores every hyperparameter defined in the search space.

Only the hyperparameters corresponding to the selected architecture are used to build the final model.

---

## Why retrain the best model?

Because the tuning process is intended for comparison. The final model should be trained completely using the selected hyperparameters.

---

## Why use EarlyStopping during tuning?

To avoid wasting time training poor-performing models.

---

## Why use ModelCheckpoint only during final training?

Because only the final model needs to be saved permanently.

---

# Summary

In this chapter, we optimized our FCNN using Keras Tuner with Bayesian Optimization. We defined a search space for multiple hyperparameters, allowed the tuner to build different architectures, and selected the best-performing configuration based on validation performance.

After retrieving the best hyperparameters, we rebuilt and retrained the model to obtain the final version used for evaluation.

---

# Key Takeaways

- Hyperparameters are chosen before training.
- Keras Tuner automates the search process.
- Bayesian Optimization intelligently selects promising trials.
- The search space defines all possible hyperparameter values.
- Dynamic architectures allow different numbers of hidden layers.
- Inactive hyperparameters remain stored but are not used.
- EarlyStopping speeds up tuning.
- The best model is rebuilt and retrained before evaluation.
