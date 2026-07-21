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

During training, the outputs of one layer continuously change because the weights are updated after every batch.

As a result, the next layer keeps receiving inputs with different distributions.

This makes training unstable.

Batch Normalization stabilizes these values.

Benefits include

- Faster convergence
- Stable gradients
- Higher learning rates
- Better generalization
- Reduced overfitting

---

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
