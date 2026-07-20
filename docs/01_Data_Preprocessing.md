# Data Preprocessing

## Overview

Data preprocessing is the first step in any deep learning pipeline. Neural networks expect numerical data in an appropriate format for efficient learning. Before training a model, the raw Fashion MNIST dataset must be transformed into a representation that is suitable for the network.

For this project, preprocessing consists of:

- Loading the Fashion MNIST dataset
- Understanding the dataset structure
- Normalizing image pixel values
- One-Hot Encoding the class labels

Proper preprocessing improves training stability, accelerates convergence, and ensures compatibility with the selected loss function.

---

# 1. Loading the Fashion MNIST Dataset

TensorFlow provides the Fashion MNIST dataset through the `keras.datasets` module.

```python
import tensorflow as tf

data = tf.keras.datasets.fashion_mnist.load_data()
```

The `load_data()` function downloads the dataset (if not already available locally) and returns the training and testing datasets.

---

# 2. Understanding the Returned Object

Before using the dataset, it is important to understand what the function returns.

```python
type(data)
```

Output

```python
<class 'tuple'>
```

The returned tuple contains two elements:

```python
(
    (X_train, y_train),
    (X_test, y_test)
)
```

where

- `X_train` → Training images
- `y_train` → Training labels
- `X_test` → Testing images
- `y_test` → Testing labels

Dataset hierarchy

```text
data
│
├── data[0]
│      ├── X_train
│      └── y_train
│
└── data[1]
       ├── X_test
       └── y_test
```

---

# 3. Exploring the Dataset

Instead of directly training the model, we first inspect the dataset.

```python
X_train.shape
```

Output

```python
(60000, 28, 28)
```

Meaning:

- 60,000 training images
- Height = 28 pixels
- Width = 28 pixels

Similarly,

```python
y_train.shape
```

Output

```python
(60000,)
```

Each image has exactly one corresponding label.

The testing dataset contains

```python
X_test.shape
```

```python
(10000, 28, 28)
```

```python
y_test.shape
```

```python
(10000,)
```

---

# 4. Image Normalization

The original Fashion MNIST images are stored as unsigned 8-bit integers (`uint8`).

Each pixel value lies between

```text
0 → Black

255 → White
```

Neural networks perform better when input features have a consistent numerical scale.

Normalization converts the pixel values into the range

```text
0.0 → Black

1.0 → White
```

using

```python
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
```

---

## Why Normalize Images?

Normalization offers several advantages:

- Faster convergence during training
- More stable gradient updates
- Improved numerical stability
- Better optimization performance

Without normalization, large input values can slow down gradient-based optimization.

---

## Why Divide by 255?

Since the maximum possible pixel value is 255,

```python
pixel / 255
```

scales every pixel into the interval

```text
[0,255]

↓

[0,1]
```

Examples

| Original | Normalized |
|----------:|-----------:|
| 0 | 0.0 |
| 128 | 0.502 |
| 255 | 1.0 |

---

## Why Convert to float32?

The original images are stored as `uint8`.

Deep learning libraries perform mathematical operations using floating-point numbers.

Converting to `float32`

```python
astype("float32")
```

provides

- Lower memory usage than float64
- Faster computation
- Better GPU compatibility
- TensorFlow's default numerical precision

---

# 5. One-Hot Encoding

The Fashion MNIST labels are integer class IDs.

Example

```text
5
```

Since the model uses a Softmax output layer together with **Categorical Crossentropy**, labels must be converted into one-hot vectors.

```python
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
```

Example

```text
5

↓

[0,0,0,0,0,1,0,0,0,0]
```

---

## Why One-Hot Encoding?

Neural networks produce a probability distribution over all classes.

Example

```text
[0.01,
0.03,
0.05,
...
0.84]
```

Categorical Crossentropy compares this probability vector against the true one-hot encoded vector.

Without one-hot encoding, the loss function cannot directly compare predicted probabilities with the target labels.

---

## Why Normalize Images but Not Labels?

Images contain numerical feature values.

Labels represent categories.

Image

```text
245
123
89
```

These values carry numerical meaning.

Labels

```text
0
1
2
...
9
```

These numbers are identifiers, not quantities.

Normalizing labels would change their meaning.

Instead, labels are converted into categorical vectors using One-Hot Encoding.

---

## Sparse Categorical Crossentropy vs Categorical Crossentropy

| Sparse Categorical Crossentropy | Categorical Crossentropy |
|---------------------------------|--------------------------|
| Integer labels | One-Hot labels |
| No encoding required | Requires One-Hot Encoding |
| Memory efficient | More intuitive for understanding probability distributions |

Since this project uses **Categorical Crossentropy**, One-Hot Encoding is required.

---

# Final Preprocessed Dataset

After preprocessing,

Images

```text
Shape

(60000,28,28)

Type

float32

Range

0.0 – 1.0
```

Labels

```text
Shape

(60000,10)

Type

float32

Representation

One-Hot Encoded
```

The dataset is now ready for building the Fully Connected Neural Network.

---

# Interview Perspective

### Why do we normalize images?

Normalization scales pixel values from `[0,255]` to `[0,1]`, improving optimization, training stability, and convergence speed.

---

### Why is `float32` preferred over `float64`?

`float32` requires less memory, performs faster computations, and is the default precision used by TensorFlow.

---

### Why are labels One-Hot Encoded?

Categorical Crossentropy expects target labels in one-hot encoded format so they can be compared directly with the probability distribution generated by the Softmax layer.

---

### Why don't we normalize labels?

Labels are categorical identifiers rather than numerical measurements. Normalization would distort their meaning.

---

### What is the difference between Sparse Categorical Crossentropy and Categorical Crossentropy?

Sparse Categorical Crossentropy accepts integer labels directly, whereas Categorical Crossentropy requires one-hot encoded labels.

---

# Key Takeaways

- Data preprocessing prepares raw data for neural network training.
- Fashion MNIST is returned as a tuple containing training and testing datasets.
- Images are normalized to improve optimization.
- `float32` is the preferred data type for deep learning.
- Labels are One-Hot Encoded because this project uses Categorical Crossentropy.
- Proper preprocessing improves training performance and model stability.
