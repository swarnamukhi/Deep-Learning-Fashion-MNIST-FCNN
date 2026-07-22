# 08 - Internal Working of a Fully Connected Neural Network (FCNN)

---

# Overview

In the previous chapters, we learned about data preprocessing, model architecture, compilation, training, hyperparameter tuning, and evaluation separately.

In this chapter, we connect all these concepts together and follow the complete journey of an image through the neural network—from the moment it is loaded until the final prediction is made.

---

# Complete FCNN Workflow

```
Input Image

↓

Data Preprocessing

↓

Flatten Layer

↓

Dense Layer (128)

↓

Batch Normalization

↓

ReLU Activation

↓

Dense Layer (64)

↓

Batch Normalization

↓

ReLU Activation

↓

Output Layer (10)

↓

Softmax

↓

Prediction

↓

Loss Calculation

↓

Backpropagation

↓

Gradient Calculation

↓

Optimizer (SGD)

↓

Weight Update

↓

Next Mini-Batch
```

---

# Step 1 – Input Image

The Fashion MNIST dataset consists of grayscale images.

Each image has a size of

```
28 × 28
```

Pixel values range from

```
0 → 255
```

Example

```
Input Shape

(28,28)
```

At this stage, the neural network only receives raw pixel values.

---

# Step 2 – Data Preprocessing

Before training, the data is prepared.

### Normalization

Pixel values

```
0 → 255
```

are converted into

```
0.0 → 1.0
```

using

```python
train_images = train_images.astype("float32") / 255
```

Normalization improves numerical stability and helps the model converge faster during training.

---

### One-Hot Encoding

The class labels are converted into one-hot vectors.

Example

```
Sneaker

↓

[0,0,0,0,0,0,0,1,0,0]
```

This format is required because the model uses the **Softmax** activation function together with the **Categorical Crossentropy** loss function.

---

# Step 3 – Flatten Layer

The neural network expects a one-dimensional input.

The Flatten layer reshapes each image.

```
28 × 28

↓

784
```

Example

```
Input

(28,28)

↓

Output

(784)
```

Important

- No learning occurs.
- No parameters are created.
- The image values remain unchanged.
- Only the shape changes.

---

# Step 4 – First Dense Layer

The flattened vector is passed to the first hidden layer.

```
784 Inputs

↓

128 Neurons
```

Each neuron performs

\[
z = W x + b
\]

where

- **W** = Weights
- **x** = Input values
- **b** = Bias

The output of every neuron is called a **logit**.

---

# Step 5 – Batch Normalization

The outputs of the Dense layer may have different scales.

Batch Normalization stabilizes these outputs.

Training Workflow

```
Dense Output

↓

Batch Mean

↓

Batch Variance

↓

Normalize

↓

Gamma Scaling

↓

Beta Shifting
```

During training

- Mean and variance are calculated from the **current mini-batch**.
- Gamma (γ) and Beta (β) are trainable parameters.
- Moving Mean and Moving Variance are updated for use during inference.

Benefits

- Faster convergence
- Stable gradients
- Higher learning rates
- Reduced sensitivity to initialization

---

# Step 6 – ReLU Activation

The normalized outputs are passed through the ReLU activation function.

Formula

\[
ReLU(x)=\max(0,x)
\]

Example

```
Input

[-3,-1,2,5]

↓

Output

[0,0,2,5]
```

ReLU introduces non-linearity, allowing the network to learn complex patterns.

---

# Step 7 – Second Hidden Layer

The activated outputs are passed to the second Dense layer.

```
128

↓

64
```

Again, each neuron computes

\[
z = Wx+b
\]

followed by

- Batch Normalization
- ReLU Activation

This layer learns higher-level features from the representations produced by the first hidden layer.

---

# Step 8 – Output Layer

The final hidden layer is connected to the output layer.

```
64

↓

10
```

Each output neuron corresponds to one Fashion MNIST class.

Examples

- T-shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

The output layer produces **logits**.

---

# Step 9 – Softmax Activation

The logits are converted into probabilities.

Example

```
Logits

[2.1,1.4,0.8,...]

↓

Softmax

↓

[0.72,
0.15,
0.05,
...]
```

Properties

- Values lie between 0 and 1.
- The probabilities sum to 1.
- The class with the highest probability becomes the predicted class.

---

# Step 10 – Loss Calculation

The predicted probabilities are compared with the true labels.

```
Prediction

↓

Categorical Crossentropy

↓

Loss
```

A lower loss indicates that the predicted probability distribution is closer to the true distribution.

The goal of training is to minimize this loss.

---

# Step 11 – Backpropagation

Once the loss is calculated, TensorFlow performs backpropagation.

The gradients of the loss with respect to every trainable parameter are computed.

```
Loss

↓

Output Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Input
```

This process determines how much each weight contributed to the prediction error.

---

# Step 12 – Optimizer (SGD)

The gradients are passed to the optimizer.

```
Gradients

↓

SGD

↓

Update Weights
```

The optimizer updates

- Dense layer weights
- Dense layer biases
- BatchNorm Gamma
- BatchNorm Beta

using the selected learning rate.

This completes one training iteration.

---

# Step 13 – Next Mini-Batch

The same sequence is repeated for every mini-batch.

Example

```
Training Samples

60000

Batch Size

32

↓

1875 Mini-Batches

↓

1 Epoch
```

After all mini-batches are processed, one epoch is complete.

The process repeats until all epochs finish.

---

# Step 14 – Validation

At the end of every epoch, the model is evaluated using the validation dataset.

```
Validation Images

↓

Forward Pass

↓

Validation Loss

↓

Validation Accuracy
```

Important

- No gradients are computed.
- No weights are updated.
- Validation is used only to measure performance on unseen data.

---

# Step 15 – Best Model

During training

- EarlyStopping monitors validation performance.
- ModelCheckpoint saves the best-performing model.

After training, the model with the lowest validation loss is retained.

---

# Step 16 – Testing

Once training is complete, the model is evaluated on the test dataset.

```
Test Image

↓

Forward Pass

↓

Softmax

↓

Prediction
```

Testing performs only inference.

No learning takes place.

---

# End-to-End Workflow

```
Input Image

↓

Normalize Pixel Values

↓

Flatten

↓

Dense (128)

↓

Batch Normalization

↓

ReLU

↓

Dense (64)

↓

Batch Normalization

↓

ReLU

↓

Dense (10)

↓

Softmax

↓

Prediction

↓

Categorical Crossentropy

↓

Loss

↓

Backpropagation

↓

Gradient Calculation

↓

SGD Optimizer

↓

Weight Update

↓

Next Mini-Batch

↓

Next Epoch

↓

Validation

↓

Best Model Saved

↓

Testing

↓

Final Prediction
```

---

# Interview Questions

## 1. Explain the complete working of an FCNN.

An FCNN receives an input image, preprocesses it, flattens it into a one-dimensional vector, passes it through multiple Dense layers with Batch Normalization and ReLU activations, generates class probabilities using Softmax, computes the loss, performs backpropagation, updates the trainable parameters using an optimizer, and repeats this process for each mini-batch until training is complete.

---

## 2. Which layer performs the first learning?

The first **Dense layer** performs the first learning by multiplying the input with trainable weights and adding biases.

---

## 3. Does the Flatten layer learn?

No. It only reshapes the input from two dimensions to one dimension. It has no trainable parameters.

---

## 4. Why is Batch Normalization applied before ReLU?

Batch Normalization stabilizes the outputs of the Dense layer before they pass through the activation function, leading to more stable and efficient training.

---

## 5. What is the purpose of ReLU?

ReLU introduces non-linearity, enabling the neural network to learn complex patterns.

---

## 6. What is the purpose of Softmax?

Softmax converts logits into a probability distribution across all classes.

---

## 7. What does Backpropagation do?

Backpropagation computes the gradients of the loss with respect to every trainable parameter so the optimizer can update them.

---

## 8. When are weights updated?

Weights are updated after processing each mini-batch during training.

---

## 9. What happens during validation?

The model performs only a forward pass to compute validation loss and accuracy. No weights are updated.

---

## 10. What is the difference between training and inference?

During training, the model performs forward propagation, loss calculation, backpropagation, and weight updates. During inference (testing or prediction), only forward propagation is performed to generate predictions.

---

# Summary

This chapter presented the complete internal workflow of a Fully Connected Neural Network. We followed the journey of an image through preprocessing, feature extraction, prediction, loss computation, backpropagation, optimization, validation, and testing. Understanding this end-to-end pipeline provides a clear picture of how every component of the network works together during training and inference.

---

# Key Takeaways

- Images are normalized before entering the network.
- Flatten converts a 2D image into a 1D feature vector.
- Dense layers learn patterns using trainable weights and biases.
- Batch Normalization stabilizes activations before ReLU.
- ReLU introduces non-linearity.
- Softmax converts logits into class probabilities.
- Categorical Crossentropy measures prediction error.
- Backpropagation computes gradients.
- SGD updates trainable parameters after each mini-batch.
- Validation measures performance without updating weights.
- Testing performs only inference using the trained model.
