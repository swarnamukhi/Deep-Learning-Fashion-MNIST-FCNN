# 07 - FCNN Interview Questions and Answers

---

# Overview

This document contains frequently asked interview questions on Fully Connected Neural Networks (FCNNs), TensorFlow, Keras, model training, hyperparameter tuning, and model evaluation.

The questions range from beginner to intermediate level and are intended to strengthen conceptual understanding.

---

# Section 1 – Neural Network Basics

## Q1. What is an Artificial Neural Network (ANN)?

An Artificial Neural Network (ANN) is a machine learning model inspired by the human brain. It consists of interconnected neurons organized into layers that learn patterns from data by adjusting weights and biases during training.

---

## Q2. What is a Fully Connected Neural Network (FCNN)?

In a Fully Connected Neural Network (FCNN), every neuron in one layer is connected to every neuron in the next layer.

Example

```
784

↓

128

↓

64

↓

10
```

Each connection has its own trainable weight.

---

## Q3. What are the main layers of an FCNN?

- Input Layer
- Hidden Layer(s)
- Output Layer

---

## Q4. What is the purpose of the input layer?

The input layer receives the feature values and passes them to the first hidden layer.

---

## Q5. What is the purpose of hidden layers?

Hidden layers learn meaningful patterns and relationships from the input data.

---

## Q6. What is the output layer?

The output layer generates the final prediction.

For Fashion MNIST

```
10 neurons

↓

10 classes
```

---

# Section 2 – Data Preprocessing

## Q7. Why do we normalize data?

Normalization scales the input values to a smaller range (typically 0–1), improving numerical stability and speeding up training.

---

## Q8. Why divide by 255?

Fashion MNIST pixel values range from 0 to 255.

Dividing by 255 converts them into values between 0 and 1.

---

## Q9. Why convert images to float32?

TensorFlow performs mathematical operations more efficiently using floating-point values.

---

## Q10. Why use One-Hot Encoding?

One-Hot Encoding converts class labels into vectors that are compatible with the Softmax activation and Categorical Crossentropy loss.

---

# Section 3 – FCNN Architecture

## Q11. What is the purpose of the Flatten layer?

Flatten converts a two-dimensional image into a one-dimensional vector without changing the data values.

---

## Q12. Does Flatten have trainable parameters?

No.

Flatten only reshapes the data.

---

## Q13. What happens inside a Dense layer?

Each neuron computes

```
z = Wx + b
```

where

- W = weights
- x = input
- b = bias

The result is then passed through an activation function.

---

## Q14. Why use ReLU?

ReLU introduces non-linearity and helps avoid the vanishing gradient problem.

Formula

```
ReLU(x)=max(0,x)
```

---

## Q15. Why not use ReLU in the output layer?

ReLU does not produce probabilities.

For multiclass classification, Softmax is more appropriate because it outputs a probability distribution.

---

## Q16. What is Softmax?

Softmax converts logits into probabilities whose sum equals 1.

---

## Q17. What are logits?

Logits are the raw outputs of the final Dense layer before the Softmax activation is applied.

---

## Q18. Why use Batch Normalization?

Batch Normalization stabilizes the outputs of a layer, leading to faster and more stable training.

---

## Q19. What are Gamma and Beta?

Gamma (γ) scales the normalized values, while Beta (β) shifts them. Both are trainable parameters.

---

## Q20. Does Batch Normalization update weights?

No.

It updates only its own trainable parameters (Gamma and Beta). The Dense layer weights are updated separately by the optimizer.

---

# Section 4 – Model Compilation

## Q21. Why use model.compile()?

It configures the optimizer, loss function, and evaluation metrics required for training.

---

## Q22. Does compile() train the model?

No.

Training begins only when model.fit() is called.

---

## Q23. What is an optimizer?

An optimizer updates the trainable parameters using gradients computed during backpropagation.

---

## Q24. What is SGD?

Stochastic Gradient Descent updates model parameters after processing each mini-batch.

---

## Q25. What is the learning rate?

The learning rate determines the size of each weight update during optimization.

---

## Q26. Why use Categorical Crossentropy?

Because Fashion MNIST is a multiclass classification problem with one-hot encoded labels.

---

# Section 5 – Model Training

## Q27. What is an epoch?

One complete pass through the training dataset.

---

## Q28. What is a batch?

A subset of the training data processed together.

---

## Q29. What is an iteration?

One forward and backward pass for a single mini-batch.

---

## Q30. When are weights updated?

After processing each mini-batch.

---

## Q31. What is forward propagation?

The process of passing inputs through the network to generate predictions.

---

## Q32. What is backpropagation?

The process of computing gradients of the loss with respect to trainable parameters.

---

## Q33. What is gradient descent?

An optimization algorithm that updates parameters in the direction that reduces the loss.

---

## Q34. What is validation data?

A separate dataset used to evaluate the model during training without updating weights.

---

## Q35. Why use EarlyStopping?

To stop training when validation performance no longer improves, helping prevent overfitting.

---

## Q36. Why use ModelCheckpoint?

To save the best-performing model during training.

---

# Section 6 – Hyperparameter Tuning

## Q37. What is a hyperparameter?

A configuration value chosen before training begins, such as learning rate or number of layers.

---

## Q38. What is Keras Tuner?

A library that automates the search for optimal hyperparameter values.

---

## Q39. Why use Bayesian Optimization?

It intelligently selects promising hyperparameter combinations instead of testing all possibilities.

---

## Q40. What is a search space?

The range of values that the tuner is allowed to explore.

---

## Q41. Why retrain the best model?

The tuning process compares models. The selected configuration is then trained fully to produce the final model.

---

# Section 7 – Model Evaluation

## Q42. What is model.evaluate()?

Evaluates the model on unseen data and returns loss and accuracy.

---

## Q43. What is model.predict()?

Generates prediction probabilities for new inputs.

---

## Q44. Why use np.argmax()?

To convert probability distributions into predicted class labels.

---

## Q45. What is a confusion matrix?

A table that summarizes correct and incorrect predictions for each class.

---

## Q46. What is precision?

Precision measures how many predicted positive samples are actually correct.

---

## Q47. What is recall?

Recall measures how many actual positive samples are correctly identified.

---

## Q48. What is the F1-score?

The harmonic mean of precision and recall.

---

## Q49. What is support?

The number of actual samples belonging to each class.

---

## Q50. Why analyze misclassified images?

To understand where the model struggles and identify opportunities for improvement.

---

# Summary

These interview questions cover the complete lifecycle of a Fully Connected Neural Network, from data preprocessing and architecture design to training, hyperparameter tuning, and evaluation. Understanding these concepts provides a strong foundation for technical interviews involving TensorFlow and deep learning.

