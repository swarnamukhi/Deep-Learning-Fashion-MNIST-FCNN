# 02 - Fully Connected Neural Network (FCNN) Architecture

---

# Overview

A **Fully Connected Neural Network (FCNN)** is one of the simplest and most fundamental neural network architectures. It is often the first deep learning model used for solving classification and regression problems.

In an FCNN, **every neuron in one layer is connected to every neuron in the next layer**. Because of these complete connections, the model can learn complex relationships between input features and output classes.

For the Fashion MNIST classification problem, the FCNN receives an image as input and predicts which clothing category it belongs to.

The overall architecture used in this project is:

```
Input Image (28 × 28)

        │

        ▼

Flatten Layer

        │

        ▼

Dense Layer (128)

        │

        ▼

Batch Normalization

        │

        ▼

ReLU Activation

        │

        ▼

Dense Layer (64)

        │

        ▼

Batch Normalization

        │

        ▼

ReLU Activation

        │

        ▼

Dense Layer (10)

        │

        ▼

Softmax

        │

        ▼

Predicted Class
```

---

# Why Do We Need an FCNN?

Images are simply collections of numbers.

For example, every Fashion MNIST image has a size of:

```
28 × 28
```

which means

```
784 pixel values
```

A neural network cannot understand that these numbers represent shoes, shirts, or bags.

Instead, it learns mathematical relationships between these pixel values and the correct output labels.

An FCNN learns these relationships by repeatedly adjusting millions of mathematical parameters during training.

Eventually, the network discovers patterns that allow it to distinguish one clothing category from another.

---

# Components of the Architecture

Our network consists of the following layers:

| Layer | Purpose |
|--------|----------|
| Flatten | Converts a 2D image into a 1D vector |
| Dense (128) | Learns high-level image features |
| Batch Normalization | Stabilizes learning |
| ReLU | Introduces non-linearity |
| Dense (64) | Learns more abstract representations |
| Batch Normalization | Stabilizes activations |
| ReLU | Adds non-linearity |
| Dense (10) | Produces one score for each class |
| Softmax | Converts scores into probabilities |

---

# Input Layer

The Fashion MNIST dataset contains grayscale images.

Each image has the shape:

```python
(28, 28)
```

During training we have

```python
train_images.shape
```

Output

```python
(60000, 28, 28)
```

which means

- 60,000 training images
- Height = 28 pixels
- Width = 28 pixels

Each pixel contains an intensity value between

```
0

and

255
```

Example

```
[[  0  15 120 ...]
 [255 230  12 ...]
 ...
]
```

After normalization

```python
train_images = train_images.astype("float32") / 255.0
```

the pixel values become

```
0.0

to

1.0
```

These normalized values become the input to the neural network.

---

# Why Do We Need the Flatten Layer?

The Dense layer expects a one-dimensional vector.

However, our images are stored as two-dimensional matrices.

Original image

```
28 × 28
```

Dense layers cannot directly process this structure.

Therefore, before sending the image into the network, we reshape it into a single vector.

This is the job of the Flatten layer.

---

# Flatten Layer

Implementation

```python
Flatten(input_shape=(28, 28))
```

Transformation

```
Before Flatten

(60000, 28, 28)

↓

After Flatten

(60000, 784)
```

Notice that

```
28 × 28 = 784
```

Every image is converted independently.

Image 1

```
28 × 28

↓

784
```

Image 2

```
28 × 28

↓

784
```

...

Image 60000

```
28 × 28

↓

784
```

The number of training samples never changes.

Only the shape of each sample changes.

---

# Internal Working of Flatten

Flatten **does not perform any calculations**.

It does not

- learn weights
- learn biases
- modify pixel values
- perform feature extraction

Instead, it simply rearranges the existing values.

For example

```
Image

1 2 3

4 5 6

7 8 9
```

becomes

```
1

2

3

4

5

6

7

8

9
```

No information is lost.

No information is created.

Only the shape changes.

---

# Trainable Parameters of Flatten

Flatten contains

```
Zero
```

trainable parameters.

TensorFlow reports

```
Flatten

Parameters = 0
```

because it only reshapes the input tensor.

---

# Dense Layer

After Flatten, every image becomes a vector of

```
784 values
```

These values are passed to the first Dense layer.

Implementation

```python
Dense(128)
```

This means

```
128 neurons
```

will be created.

Each neuron receives

```
all 784 input values
```

This is why the layer is called

```
Fully Connected
```

Every neuron is connected to every input feature.

---

# Internal Working of a Neuron

Each neuron performs two operations.

Step 1

Weighted Sum

```
z = w₁x₁ + w₂x₂ + w₃x₃ + ... + w₇₈₄x₇₈₄ + b
```

where

- x = input values
- w = learnable weights
- b = bias

The output of this calculation is called

```
z
```

Step 2

Activation Function

The weighted sum is passed through an activation function.

```
Output = Activation(z)
```

This output becomes the input to the next layer.

---

# Why Do We Need Dense Layers?

Flatten simply rearranges the image.

It does not learn anything.

The Dense layers are responsible for learning meaningful patterns.

For example

- edges
- brightness combinations
- texture patterns
- clothing-specific features

As training progresses, the weights inside the Dense layers are updated so that the model becomes better at distinguishing different clothing categories.

---

# Key Takeaways
---

# Activation Functions

A Dense layer computes a weighted sum of its inputs.

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

However, this weighted sum alone is **not sufficient** to solve complex problems.

If we stack multiple Dense layers without any activation function, the entire neural network behaves like a single linear equation, regardless of how many layers we add.

To overcome this limitation, we apply an **activation function** after each Dense layer.

The activation function introduces **non-linearity**, allowing the neural network to learn complex relationships within the data.

In this project, we use two activation functions:

- ReLU (Hidden Layers)
- Softmax (Output Layer)

---

# ReLU (Rectified Linear Unit)

The Rectified Linear Unit (ReLU) is the most widely used activation function in deep learning.

It is defined as

```
ReLU(x) = max(0, x)
```

This means:

```
Input < 0

↓

Output = 0
```

```
Input > 0

↓

Output = Input
```

Examples

| Input | Output |
|--------|---------|
| -5 | 0 |
| -2 | 0 |
| 0 | 0 |
| 3 | 3 |
| 8 | 8 |

---

# Why Do We Need ReLU?

Suppose a Dense layer produces

```
[-12, -3, 5, 10]
```

After applying ReLU

```
[0, 0, 5, 10]
```

All negative values become zero.

Positive values remain unchanged.

This introduces **non-linearity**, enabling the neural network to learn complex decision boundaries instead of only linear relationships.

Without ReLU, even a deep network with many Dense layers behaves like a single linear transformation.

---

# Why Does Removing Negative Values Introduce Non-Linearity?

This is one of the most common interview questions.

Consider the graph of the function

```
y = x
```

This is a straight line.

A straight line is called a **linear function**.

Now consider ReLU.

```
For x < 0

Output = 0

For x > 0

Output = x
```

The graph now has a bend at zero.

```
Negative Side

──────────────

0

/

/

/

Positive Side
```

Because the graph is no longer a single straight line, ReLU is called a **non-linear activation function**.

This non-linearity enables deep neural networks to model highly complex real-world data.

---

# Why Is ReLU Used Instead of Sigmoid?

ReLU offers several advantages.

- Faster computation
- Simpler mathematical operation
- Reduces the vanishing gradient problem
- Speeds up training
- Produces sparse activations

For these reasons, ReLU has become the default activation function for hidden layers.

---

# Softmax Activation

The final Dense layer produces **raw scores**, also called **logits**.

Example

```
[3.2, 0.5, 1.7, 4.1, 2.6, 1.0, 0.4, 0.8, 2.1, 3.5]
```

These values are **not probabilities**.

Some values may even be negative.

Therefore, we apply the Softmax activation.

```
Dense(10)

↓

Softmax

↓

Probability Distribution
```

Softmax converts the logits into probabilities between

```
0

and

1
```

The probabilities always satisfy

```
Total Sum = 1
```

Example

```
[0.01,
0.03,
0.05,
0.48,
0.08,
0.02,
0.01,
0.01,
0.06,
0.25]
```

Notice

```
All values

≥ 0

and

Total = 1
```

The class with the highest probability becomes the predicted class.

---

# Why Softmax?

Our Fashion MNIST dataset has

```
10 classes
```

The model must decide

```
Which one is most likely?
```

Softmax converts arbitrary output scores into a valid probability distribution, making it easy to interpret the model's prediction.

---

# Parameter Calculation

Understanding parameter calculation is extremely important for interviews.

TensorFlow displays the number of trainable parameters in the model summary.

Let's calculate them manually.

---

## Dense Layer (128)

Input Features

```
784
```

Neurons

```
128
```

Each neuron has

- 784 weights
- 1 bias

Total

```
(784 × 128) + 128

=

100352 + 128

=

100480
```

---

## Dense Layer (64)

Previous Layer

```
128 outputs
```

Current Layer

```
64 neurons
```

Parameters

```
(128 × 64) + 64

=

8192 + 64

=

8256
```

---

## Output Layer

Input

```
64
```

Output Neurons

```
10
```

Parameters

```
(64 × 10) + 10

=

640 + 10

=

650
```

---

# Total Trainable Parameters

```
100480

+

8256

+

650

=

109386
```

This exactly matches the TensorFlow model summary.

---

# Batch Normalization

Batch Normalization is one of the most important improvements in deep learning.

It normalizes the outputs of a layer before they are passed to the activation function.

Architecture

```
Dense

↓

Batch Normalization

↓

ReLU
```

---
# Why Do We Need Batch Normalization?

During training, the neural network processes the dataset one **mini-batch** at a time.

Suppose our model uses

- Image Size = 28 × 28
- Flatten = 784 features
- Batch Size = 32

Then each mini-batch entering the first Dense layer has the shape

```
(32, 784)
```

During the forward pass, these 32 samples pass through the first Dense layer.

```
Batch (32,784)

↓

Dense(128)

↓

Output Shape = (32,128)
```

After the forward pass, the model computes the loss and performs backpropagation.

As a result, the **trainable parameters of the entire network are updated**, including

- Dense layer weights
- Dense layer biases
- Batch Normalization Gamma (γ)
- Batch Normalization Beta (β)

When the next mini-batch arrives, the network now has **updated weights**.

Therefore, even if the new batch contains similar images, the outputs (also called **activations**) produced by the Dense layer will be different because the weights have changed.

Without Batch Normalization, every layer would continuously receive inputs with changing distributions throughout training, making learning less stable.

Batch Normalization reduces this variation by normalizing the activations of **each mini-batch** before passing them to the next layer.

This results in

- More stable training
- Faster convergence
- Better gradient flow
- Improved generalization

---

# Internal Working of Batch Normalization

Consider the first hidden layer.

After the Dense layer, the output shape is

```
(32,128)
```

which means

- 32 samples
- 128 neuron outputs (features)

Batch Normalization now processes these activations.

## Step 1: Compute Batch Statistics

For the current mini-batch, Batch Normalization calculates

- Mean (μ)
- Variance (σ²)

using only the activations of the current batch.

For example

```
Batch 1

↓

Mean = 18

Variance = 49
```

These values are computed independently for each of the 128 features.

---

## Step 2: Normalize

Each activation is normalized using the batch statistics.

After normalization

```
Mean ≈ 0

Standard Deviation ≈ 1
```

This ensures that the activations are on a consistent scale before being passed to the next layer.

---

## Step 3: Learn the Best Scale and Shift

Instead of always forcing the output to have Mean = 0 and Standard Deviation = 1, Batch Normalization introduces two learnable parameters.

- Gamma (γ) → controls the scale
- Beta (β) → controls the shift

The final output becomes

```
Output

=

γ × Normalized Value

+

β
```

Both Gamma and Beta are updated during backpropagation just like the weights of the Dense layers.

---

## What Happens for the Next Mini-Batch?

After Batch 1 finishes,

```
Forward Pass

↓

Loss Calculation

↓

Backpropagation

↓

Update All Trainable Parameters
```

Now Batch 2 enters the network.

Because the weights have changed, the Dense layer produces different activations.

For example

```
Batch 1

↓

Dense Output

Mean = 18
```

After the weight update

```
Batch 2

↓

Dense Output

Mean = 42
```

Batch Normalization does **not** reuse the statistics from Batch 1.

Instead, it computes **new Mean and Variance** for Batch 2 and normalizes that batch independently.

This process repeats for every mini-batch during training.

---

# Important Note

A common misconception is that Batch Normalization uses the statistics from the previous batch to normalize the next batch.

This is **not correct**.

During training,

- Batch 1 is normalized using Batch 1 statistics.
- Batch 2 is normalized using Batch 2 statistics.
- Batch 3 is normalized using Batch 3 statistics.

Each mini-batch is normalized independently.

The moving mean and moving variance maintained by Batch Normalization are **not used during training**. They are stored and later used during **inference (prediction/testing)**, where batch statistics may not be available.



# Internal Working

For every mini-batch, Batch Normalization performs

Step 1

Compute batch mean

```
μ
```

Step 2

Compute batch variance

```
σ²
```

Step 3

Normalize

```
Mean = 0

Standard Deviation = 1
```

Step 4

Scale

```
γ × normalized_value
```

Step 5

Shift

```
+ β
```

Final Output

```
Output

=

γ × normalized_value

+

β
```

---

# Why Gamma (γ) and Beta (β)?

If every layer always produced outputs with

```
Mean = 0

Standard Deviation = 1
```

the network would lose flexibility.

Gamma and Beta allow the network to learn the most useful output distribution.

- Gamma controls the scale.
- Beta controls the shift.

Both are trainable parameters.

---

# Trainable Parameters of Batch Normalization

For every feature, Batch Normalization stores

- Gamma (γ)
- Beta (β)
- Moving Mean
- Moving Variance

Therefore

```
Parameters

=

4 × Number of Features
```

For

```
BatchNormalization(128)
```

```
128 × 4

=

512
```

For

```
BatchNormalization(64)
```

```
64 × 4

=

256
```

Among these,

- Gamma and Beta are trainable.
- Moving Mean and Moving Variance are non-trainable.

---

# Complete Architecture

```
Input (28×28)

↓

Flatten

↓

Dense(128)

↓

Batch Normalization

↓

ReLU

↓

Dense(64)

↓

Batch Normalization

↓

ReLU

↓

Dense(10)

↓

Softmax

↓

Prediction
```

---

# Key Takeaways

- ReLU introduces non-linearity.
- Softmax converts logits into probabilities.
- Dense layers contain trainable weights and biases.
- Parameter calculation follows the formula:

```
(Input Features × Neurons) + Biases
```

- Batch Normalization stabilizes training.
- Gamma and Beta are learnable parameters.
- Moving Mean and Moving Variance are used during inference.
- The complete FCNN contains **109,386 trainable parameters**.

- FCNN is one of the simplest neural network architectures.
- Every neuron is connected to every neuron in the next layer.
- Flatten converts images into vectors.
- Flatten has **zero trainable parameters**.
- Dense layers learn patterns by adjusting weights and biases.
- Every neuron computes a weighted sum followed by an activation function.
- Learning happens only inside the Dense layers.

---

# TensorFlow/Keras Implementation

The following code implements the Fully Connected Neural Network (FCNN) used in this project.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation

model = Sequential([
    Flatten(input_shape=(28, 28)),

    Dense(128),
    BatchNormalization(),
    Activation("relu"),

    Dense(64),
    BatchNormalization(),
    Activation("relu"),

    Dense(10, activation="softmax")
])
```

---

# Layer-by-Layer Explanation

### Flatten Layer

```python
Flatten(input_shape=(28,28))
```

- Converts each **28 × 28** image into a **784-dimensional vector**.
- Performs only reshaping.
- Contains **0 trainable parameters**.

```
(28 × 28)

↓

784
```

---

### First Dense Layer

```python
Dense(128)
```

- Creates **128 neurons**.
- Every neuron receives all **784 input features**.
- Learns meaningful patterns by updating weights and biases during training.

Trainable Parameters

```
(784 × 128) + 128

=

100480
```

---

### Batch Normalization

```python
BatchNormalization()
```

Normalizes the activations of the **current mini-batch** before they are passed to the next layer.

Learns

- Gamma (γ)
- Beta (β)

Maintains

- Moving Mean
- Moving Variance

Benefits

- Stable training
- Faster convergence
- Better gradient flow
- Improved generalization

---

### ReLU Activation

```python
Activation("relu")
```

Applies

```
ReLU(x)=max(0,x)
```

Example

```
[-4,-2,0,5,8]

↓

[0,0,0,5,8]
```

Introduces non-linearity, enabling the network to learn complex decision boundaries.

---

### Second Dense Layer

```python
Dense(64)
```

- Receives the 128 outputs from the previous layer.
- Learns higher-level feature representations.

Trainable Parameters

```
(128 × 64) + 64

=

8256
```

---

### Output Layer

```python
Dense(10, activation="softmax")
```

Creates one output neuron for each Fashion MNIST class.

Output Shape

```
(batch_size,10)
```

Example

```
(32,10)
```

---

### Softmax Activation

Softmax converts the output scores (logits) into probabilities.

Example

```
Before Softmax

[3.2,0.5,1.8,4.1,...]

↓

After Softmax

[0.02,
0.01,
0.05,
0.71,
...]
```

Properties

- Values are between **0 and 1**
- Total probability equals **1**
- Highest probability becomes the predicted class

---

# Complete FCNN Workflow

```
Fashion MNIST Images

↓

Normalize Pixel Values

↓

Flatten

↓

Dense(128)

↓

Batch Normalization

↓

ReLU

↓

Dense(64)

↓

Batch Normalization

↓

ReLU

↓

Dense(10)

↓

Softmax

↓

Predicted Probabilities

↓

Loss Calculation

↓

Backpropagation

↓

Update Weights

↓

Next Mini-Batch
```

---

# Interview Questions

## 1. What is a Fully Connected Neural Network?

A neural network in which every neuron is connected to every neuron in the next layer.

---

## 2. Why do we use Flatten?

Flatten converts a two-dimensional image into a one-dimensional feature vector so that it can be processed by Dense layers.

---

## 3. Does Flatten learn anything?

No.

Flatten only reshapes the input tensor.

It has **zero trainable parameters**.

---

## 4. Why is the Dense layer called Fully Connected?

Because every neuron receives input from every neuron in the previous layer.

---

## 5. Why do we use ReLU?

ReLU

- Introduces non-linearity
- Speeds up training
- Reduces the vanishing gradient problem
- Is computationally efficient

---

## 6. Why is Softmax used in the output layer?

Softmax converts logits into probabilities, making it suitable for multiclass classification problems.

---

## 7. Why do we use Batch Normalization?

Batch Normalization normalizes the activations of each mini-batch, resulting in faster, more stable, and more efficient training.

---

## 8. Does Batch Normalization normalize the next batch?

No.

Each mini-batch is normalized independently using its own Mean and Variance.

---

## 9. Which Batch Normalization parameters are trainable?

- Gamma (γ)
- Beta (β)

---

## 10. Which Batch Normalization parameters are not trainable?

- Moving Mean
- Moving Variance

These are updated during training and later used during inference.

---

## 11. How many trainable parameters does this model have?

```
Dense(128)

100480

+

Dense(64)

8256

+

Dense(10)

650

=

109386
```

---

# Summary

In this chapter, we designed a Fully Connected Neural Network (FCNN) to classify Fashion MNIST images.

We explored the purpose and internal working of every layer, including Flatten, Dense, Batch Normalization, ReLU, and Softmax. We also manually calculated the trainable parameters of each Dense layer and understood how Batch Normalization stabilizes training by normalizing the activations of each mini-batch.

Finally, we implemented the complete architecture using TensorFlow/Keras and reviewed the most common interview questions related to FCNNs.

---

# Key Takeaways

- FCNN is a feedforward neural network where every neuron is connected to every neuron in the next layer.
- Flatten converts a 2D image into a 1D vector without changing the data.
- Dense layers learn patterns by updating weights and biases.
- ReLU introduces non-linearity, allowing the model to learn complex relationships.
- Softmax converts logits into a probability distribution for multiclass classification.
- Batch Normalization normalizes the activations of each mini-batch to stabilize learning.
- Gamma and Beta are trainable Batch Normalization parameters.
- Moving Mean and Moving Variance are maintained during training and used during inference.
- The FCNN used in this project contains **109,386 trainable parameters**.
