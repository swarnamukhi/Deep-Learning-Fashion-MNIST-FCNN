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

- FCNN is one of the simplest neural network architectures.
- Every neuron is connected to every neuron in the next layer.
- Flatten converts images into vectors.
- Flatten has **zero trainable parameters**.
- Dense layers learn patterns by adjusting weights and biases.
- Every neuron computes a weighted sum followed by an activation function.
- Learning happens only inside the Dense layers.
