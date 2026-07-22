# 06 - Model Evaluation

---

# Overview

After training the neural network, the next step is to evaluate its performance.

Model evaluation helps us understand how well the trained model performs on unseen data. It also helps identify strengths, weaknesses, and areas for improvement.

In this project, we evaluate the model using several techniques, including:

- Model Evaluation
- Model Prediction
- Accuracy Score
- Classification Report
- Confusion Matrix
- Misclassified Images
- Single Image Prediction

---

# Model Evaluation Workflow

```
Train Model

↓

Evaluate Model

↓

Generate Predictions

↓

Convert Probabilities to Labels

↓

Calculate Accuracy

↓

Classification Report

↓

Confusion Matrix

↓

Misclassified Images

↓

Predict New Images
```

---

# Evaluating the Model

TensorFlow provides the `evaluate()` method to measure the model's performance on unseen data.

```python
test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)
```

The method returns

- Test Loss
- Test Accuracy

Example

```
Test Loss

0.24

Test Accuracy

91.8%
```

---

# What Happens During evaluate()?

When `evaluate()` is called

```
Test Images

↓

Forward Pass

↓

Predictions

↓

Loss Calculation

↓

Accuracy Calculation

↓

Return Results
```

Important

- No Backpropagation
- No Gradient Calculation
- No Weight Updates

The model only performs inference.

---

# Model Prediction

To obtain predictions for each test image

```python
predictions = model.predict(test_images)
```

Unlike `evaluate()`, the `predict()` method requires only the input images.

Output Shape

```
(10000,10)
```

This means

- 10,000 test images
- 10 probabilities for each image

Example

```
Image 1

↓

[0.01,
0.02,
0.05,
...
0.76]
```

Each row represents the probability distribution for one image.

---

# Why Do We Use np.argmax()?

The model predicts probabilities rather than class labels.

To determine the predicted class, we select the index with the highest probability.

```python
predicted_labels = np.argmax(
    predictions,
    axis=1
)
```

Example

Prediction

```
[0.02,
0.01,
0.70,
0.10,
...
]
```

Highest Probability

```
0.70
```

Index

```
2
```

Predicted Class

```
2
```

Similarly,

```python
true_labels = np.argmax(
    test_labels,
    axis=1
)
```

converts one-hot encoded labels into class indices.

---

# Why axis=1?

Prediction Shape

```
(10000,10)
```

Rows

```
Images
```

Columns

```
Class Probabilities
```

Using

```python
axis=1
```

means

"For each image, find the class with the highest probability."

Output Shape

```
(10000,)
```

---

# Accuracy Score

We can also calculate accuracy using Scikit-learn.

```python
from sklearn.metrics import accuracy_score

accuracy_score(
    true_labels,
    predicted_labels
)
```

Accuracy measures

```
Correct Predictions

───────────────

Total Predictions
```

Example

```
9200 Correct

10000 Total

↓

92%
```

---

# evaluate() vs accuracy_score()

| evaluate() | accuracy_score() |
|------------|------------------|
| TensorFlow | Scikit-learn |
| Requires images and labels | Requires predicted and true labels |
| Returns loss and accuracy | Returns only accuracy |
| Performs forward pass | Uses existing predictions |

---

# Classification Report

The classification report provides detailed metrics for every class.

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        true_labels,
        predicted_labels
    )
)
```

Typical Output

```
precision

recall

f1-score

support
```

---

# Precision

Precision answers the question

> "When the model predicts a class, how often is it correct?"

Formula

```
TP

────────────

TP + FP
```

High precision means

Few False Positives.

---

# Recall

Recall answers the question

> "Out of all actual samples belonging to a class, how many did the model correctly identify?"

Formula

```
TP

────────────

TP + FN
```

High recall means

Few False Negatives.

---

# F1-Score

F1-score combines Precision and Recall into a single metric.

Formula

```
2 × Precision × Recall

────────────────────────

Precision + Recall
```

A higher F1-score indicates a better balance between precision and recall.

---

# Support

Support indicates the number of actual samples belonging to each class.

For Fashion MNIST

```
1000
```

samples exist for each class.

Support does not measure model performance.

It simply shows how many samples belong to that class.

---

# Macro Average

Macro Average calculates the simple average of all classes.

Each class contributes equally regardless of its size.

---

# Weighted Average

Weighted Average considers the number of samples in each class.

Classes with more samples contribute more to the final average.

Since Fashion MNIST contains equal samples for every class,

```
Macro Average

≈

Weighted Average
```

---

# Confusion Matrix

A confusion matrix summarizes how many predictions are correct and incorrect.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    true_labels,
    predicted_labels
)
```

Rows

```
Actual Classes
```

Columns

```
Predicted Classes
```

A perfect model produces a strong diagonal because correct predictions appear on the diagonal.

Off-diagonal values represent misclassifications.

---

# Misclassified Images

Misclassified images help us understand where the model makes mistakes.

```python
incorrect = np.where(
    predicted_labels != true_labels
)[0]
```

These indices can be used to visualize incorrectly classified images.

This analysis helps identify confusing classes, such as

- Shirt vs T-shirt
- Coat vs Pullover

---

# Predicting a Single Image

A trained model can also classify a single image.

```python
image = test_images[0]

prediction = model.predict(
    image.reshape(1,28,28)
)

predicted_class = np.argmax(
    prediction
)
```

The image is reshaped because the model expects a batch dimension.

```
(28,28)

↓

(1,28,28)
```

---

# Complete Evaluation Pipeline

```
Load Test Images

↓

model.evaluate()

↓

model.predict()

↓

np.argmax()

↓

accuracy_score()

↓

classification_report()

↓

confusion_matrix()

↓

Analyze Wrong Predictions

↓

Predict New Images
```

---

# Interview Questions

## What is model.evaluate()?

Evaluates the trained model on unseen data and returns loss and accuracy.

---

## Does evaluate() update weights?

No.

It performs only forward propagation.

---

## What is model.predict()?

Generates predictions for new input data.

---

## Why do we use np.argmax()?

To convert predicted probabilities into class labels.

---

## What is the difference between evaluate() and predict()?

`evaluate()` returns loss and accuracy.

`predict()` returns probability distributions.

---

## Why use accuracy_score()?

To calculate accuracy using Scikit-learn.

---

## What does Precision measure?

The proportion of predicted positives that are actually correct.

---

## What does Recall measure?

The proportion of actual positives that the model correctly identifies.

---

## What is F1-score?

The harmonic mean of Precision and Recall.

---

## What is Support?

The number of actual samples belonging to each class.

---

## Why analyze misclassified images?

To understand where the model struggles and identify opportunities for improvement.

---

# Summary

In this chapter, we evaluated the trained FCNN using TensorFlow and Scikit-learn. We learned how to measure performance using `model.evaluate()`, generate predictions with `model.predict()`, convert probabilities into class labels using `np.argmax()`, and calculate detailed performance metrics using Accuracy Score, Classification Report, and Confusion Matrix.

Finally, we analyzed misclassified images and demonstrated how to predict the class of a single image.

---

# Key Takeaways

- `model.evaluate()` measures loss and accuracy on unseen data.
- `model.predict()` generates probability distributions.
- `np.argmax()` converts probabilities into class labels.
- `accuracy_score()` calculates classification accuracy.
- Precision, Recall, and F1-score provide detailed class-wise evaluation.
- Confusion Matrix helps identify classification errors.
- Misclassified images reveal the model's weaknesses.
- Single image prediction is performed by adding a batch dimension before inference.
