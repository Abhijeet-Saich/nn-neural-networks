# House Price Prediction with PyTorch: Revision Guide

A beginner-friendly neural network that predicts California house prices, built first from raw tensors and then with PyTorch's built-in tools. This guide covers every concept used in the project, for revision and interview preparation.

---

## Table of Contents

1. [Project Summary and Results](#1-project-summary-and-results)
2. [Environment Setup](#2-environment-setup)
3. [The Dataset](#3-the-dataset)
4. [The Full Pipeline at a Glance](#4-the-full-pipeline-at-a-glance)
5. [Train/Test Split](#5-traintest-split)
6. [Feature Scaling](#6-feature-scaling)
7. [Tensors](#7-tensors)
8. [TensorDataset and DataLoader](#8-tensordataset-and-dataloader)
9. [Neural Network Fundamentals](#9-neural-network-fundamentals)
10. [Weight Initialization](#10-weight-initialization)
11. [Loss Functions: MSE and RMSE](#11-loss-functions-mse-and-rmse)
12. [The Training Loop](#12-the-training-loop)
13. [Evaluation](#13-evaluation)
14. [Hyperparameters and Experiments](#14-hyperparameters-and-experiments)
15. [Raw Code vs Built-in PyTorch](#15-raw-code-vs-built-in-pytorch)
16. [Function Reference](#16-function-reference)
17. [Bugs We Hit and Lessons Learned](#17-bugs-we-hit-and-lessons-learned)
18. [Interview Questions](#18-interview-questions)
19. [Write It Yourself: Full Code Templates](#19-write-it-yourself-full-code-templates)

---

## 1. Project Summary and Results

**Goal:** given 8 facts about a California neighborhood (income, house age, location, and so on), predict its median house value.

**Task type:** regression, meaning we predict a continuous number rather than a category.

**Model:** a feedforward neural network, 8 → 32 → 64 → 1, with ReLU activations. It has 2,465 learnable parameters.

| Measurement | Value |
|---|---|
| Loss before training (5 test rows) | 7.48 |
| Final training loss (MSE, epoch 50) | 0.2956 |
| Training error (RMSE) | $54,014 |
| **Test error (RMSE)** | **$56,109** |
| Baseline error (always guess the average) | $114,486 |

**What the results mean:**

- The model **halved the baseline's error**, so it learned real relationships between the features and price.
- Training error and test error are **close together**, so the model **generalizes** to new data rather than memorizing the training rows.

### What kind of neural network is this?

The network in this project is a **Multilayer Perceptron (MLP)**. You'll also hear it called a **feedforward neural network**, a **fully connected network**, or a **dense network**. It's the simplest and most fundamental type of neural network, and every other architecture builds on its ideas.

Each name describes one property:

- **Feedforward:** data flows in one direction only, from input to output, with no loops and no memory of previous examples. Each prediction is made independently.
- **Fully connected (dense):** every neuron in one layer connects to every neuron in the next layer. That's exactly what `nn.Linear` (or `x @ W + b`) does.
- **Multilayer:** it has hidden layers between the input and the output, and these let it learn non-linear patterns that a single layer (plain linear regression) cannot.

Our network's structure:

```
Input layer      Hidden layer 1     Hidden layer 2      Output layer
8 features  →    32 neurons    →    64 neurons     →    1 neuron
                 + ReLU             + ReLU              (no activation)
```

- **Input layer:** the 8 scaled features. It has no weights; it just holds the data.
- **Hidden layers:** 2 layers that learn intermediate features, each followed by ReLU.
- **Output layer:** 1 neuron giving the predicted price, with no activation because this is regression.

**Learning setup:** **supervised learning** (every training example comes with the correct answer) for a **regression** task (predicting a continuous number).

**Depth:** with 2 hidden layers and 2,465 parameters, this is a **shallow** network. "Deep" learning usually refers to networks with many layers, though there's no strict cutoff.

**Why an MLP fits this problem:** our data is **tabular**, meaning rows of independent numeric features with no spatial or time-order structure. MLPs are the natural first choice for this kind of data. Other architectures exist for data with structure:

| Architecture | Built for | Key idea |
|---|---|---|
| **MLP (this project)** | Tabular data | Every input connects to every neuron |
| **CNN** (Convolutional Neural Network) | Images | Small filters slide across the image to detect local patterns like edges and shapes |
| **RNN / LSTM** (Recurrent Neural Network) | Sequences, time series | Loops that carry memory from one step to the next |
| **Transformer** | Text, sequences, and more | Attention, which lets every position look at every other position |

MLPs also appear **inside** these bigger architectures. For example, a CNN often ends with a few fully connected layers, and every Transformer block contains an MLP. Understanding this project's network is the foundation for all of them.

---

## 2. Environment Setup

### What installing Python actually installs

Installing Python is like installing Node.js. You get:

| Piece | What it is | Node.js equivalent |
|---|---|---|
| Interpreter (CPython) | The program that reads and runs `.py` files | The `node` executable |
| Standard library | Built-in modules (`math`, `json`, `os`) | Built-in modules (`fs`, `path`) |
| pip | Package installer | npm |
| PyPI | Online package registry | npm registry |

Python first compiles your code into **bytecode** (the `.pyc` files in `__pycache__`), then runs it on its own virtual machine.

### Multiple Python versions

- You can install **any number of Python versions** side by side. Each lives in its own folder with its own packages.
- When you type `python3`, macOS searches the folders listed in your **PATH** in order and runs the first match. `which python3` shows which one that is.
- Version-numbered commands like `python3.12` always run that exact version.
- Leave Apple's system Python (`/usr/bin/python3`) alone.
- Tools like **pyenv** or **uv** help manage many versions.

### Virtual environments

A virtual environment (venv) is an isolated folder of packages tied to **one** Python version. The standard practice is **one venv per project**.

```bash
python3.12 -m venv .venv        # create (tied to Python 3.12)
source .venv/bin/activate       # activate
pip install torch scikit-learn numpy jupyter
deactivate                      # leave the environment
```

While a venv is active, plain `python` and `pip` refer to that environment.

### Why `pip install torch` failed

`ERROR: Could not find a version that satisfies the requirement torch (from versions: none)`

PyTorch is mostly **compiled C++ code**, published as separate builds for each combination of:

- operating system (macOS, Linux, Windows)
- chip architecture (`arm64` for Apple Silicon, `x86_64` for Intel)
- Python version

If no build matches your exact setup, pip finds nothing to install. **Fix:** use a well-supported Python version (3.12 is safe everywhere) and a native `arm64` Python on Apple Silicon. Check the architecture with:

```bash
python -c "import platform; print(platform.machine())"
```

Pure-Python packages don't have this problem, because the interpreter can run them anywhere.

### Jupyter notebook habits

- Notebooks **remember variables between cells**. Re-running a cell works on whatever the variables *currently* hold, not the original data.
- Deleted cells' variables still exist in memory and can hide bugs.
- When things act strangely, use **Kernel → Restart Kernel and Run All** for a clean slate.
- The notebook's **kernel** decides which environment, and therefore which Python, your code runs in. Point it at your project's venv.
- Putting a variable name on the last line of a cell displays it automatically.

---

## 3. The Dataset

**California Housing** ships with scikit-learn. It comes from the 1990 US Census.

- **20,640 rows.** Each row is a **census block group** (a neighborhood of a few hundred to a few thousand people), not a single house.
- **8 features (X):**

| Feature | Meaning |
|---|---|
| MedInc | Median income in the block, in tens of thousands of dollars |
| HouseAge | Median house age in years |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Number of people in the block |
| AveOccup | Average people per household |
| Latitude | North–south location |
| Longitude | East–west location |

- **Target (y):** median house value **in units of $100,000**. So 2.78 means $278,000. We multiply by 100,000 only at the end, to display errors in dollars.
- **The cap:** values are capped at **5.0 ($500,000)**. Any block worth more was recorded as exactly 5.0, so the model can't learn what very expensive blocks are truly worth.

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target
print(X.shape, y.shape)          # (20640, 8) (20640,)
print(housing.feature_names)
print(housing.DESCR)             # full description of the dataset
```

`fetch_california_housing()` returns a **Bunch** object, which works like a dictionary with attribute access (`.data`, `.target`, `.feature_names`, `.DESCR`).

---

## 4. The Full Pipeline at a Glance

```
Load data → Split train/test → Scale features → Convert to tensors
   → Batch with DataLoader → Define model → Choose loss + optimizer
   → Training loop (predict → loss → backward → update) → Evaluate on test set
```

Every supervised deep learning project follows roughly this shape.

---

## 5. Train/Test Split

**Why:** the model trains on the training set many times, so its training error is flattering, like grading a student on questions they've already practiced. The **test set** is held back as unseen data, which makes it the real exam.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

| Parameter | Meaning |
|---|---|
| `test_size=0.2` | 20% for testing (4,128 rows), 80% for training (16,512 rows) |
| `random_state=42` | Fixes the random shuffle, so the split is the same every run |

**Watch out for the return order:** `X_train, X_test, y_train, y_test`, so both X pieces come first. Python unpacks the results in order, and getting the names wrong silently puts data in the wrong variables.

X and y are shuffled **together**, so row *i* of `x_train` still matches row *i* of `y_train`.

---

## 6. Feature Scaling

### The problem

The features live on wildly different scales. Income runs from about 0.5 to 15, while population runs from about 3 to 35,000. Size doesn't mean importance, but a network can't tell:

1. Gradients scale with input size, so **large-valued features get huge weight updates and small-valued features get tiny ones**. The network overreacts to population and barely learns from income, which is actually the strongest predictor.
2. There's **one learning rate for all weights**. A step size that suits one feature overshoots on another, so no single learning rate works well.

### The fix: standardization

Convert each feature to "how many standard deviations from the average is this value?"

```
z = (x − mean) / standard_deviation
```

Example: an income of 7.67, with a mean of 3.87 and a standard deviation of 1.90, becomes (7.67 − 3.87) / 1.90 = **2.0**, meaning two standard deviations above average.

After scaling, every feature has **mean 0 and standard deviation 1**, so all features are on equal footing.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # learn mean/std from TRAIN, then scale it
x_test  = scaler.transform(x_test)        # reuse TRAIN's mean/std on test

print(x_train.mean(axis=0).round(2))      # all ≈ 0
print(x_train.std(axis=0).round(2))       # all ≈ 1
```

| Method | What it does |
|---|---|
| `fit` | Calculates each column's mean and standard deviation, and stores them |
| `transform` | Applies `(x − mean) / std` using the stored values |
| `fit_transform` | Both in one step |
| `scaler.mean_`, `scaler.scale_` | The stored means and standard deviations |
| `axis=0` | Calculate down each column (per feature), not across rows |

**`-0.` in the output** is a tiny negative rounding error (like -2e-16) shown at 2 decimals. It's mathematically zero.

### Data leakage

**Never fit the scaler on the test set.** The test set represents future, unseen data. If the scaler learns from it, test information leaks into training and your score looks better than it really is. Fit on training data only, then transform both sets.

**y is not scaled.** It stays in its original units, so results are easy to interpret.

---

## 7. Tensors

A **tensor** is PyTorch's array, like a NumPy array but able to run on a GPU and track gradients.

```python
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
```

**Why float32:** NumPy defaults to float64, but PyTorch layers expect float32. Mixing them causes type errors.

**Why `.view(-1, 1)`:** `y` starts as shape `(16512,)`, a flat list. The model outputs shape `(16512, 1)`, a column. If the shapes differ, the loss calculation **broadcasts** them into a (16512 × 16512) grid and silently produces garbage. `view(-1, 1)` means "1 column, and work out the number of rows (-1) automatically." `.reshape(-1, 1)` and `.unsqueeze(1)` do the same thing.

**Shape reading:** `(rows, columns)`. `x_train` is `(16512, 8)`: 16,512 examples with 8 features each.

**Converting a tensor that's already a tensor** with `torch.tensor(...)` triggers a warning that recommends `.clone().detach()`. In a notebook, it usually means you ran the cell twice.

---

## 8. TensorDataset and DataLoader

### TensorDataset: pairs features with targets

```python
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(x_train, y_train)
```

It staples each row of features to its price, so `train_dataset[5]` returns `(features_5, price_5)`.

### DataLoader: serves the pairs in batches

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
len(train_loader)   # 258 batches per epoch (16512 / 64)
```

| Job | Meaning |
|---|---|
| Batching | Hands over 64 pairs at a time |
| Shuffling | A new random order every epoch, so the model doesn't learn the row order |
| Iteration | `for x_batch, y_batch in train_loader:` gives one batch per pass |

- Creating a DataLoader returns a **DataLoader object**, not data. Data comes out when you loop over it.
- Each batch is a pair: `x_batch` of shape **(64, 8)** and `y_batch` of shape **(64, 1)**.
- `next(iter(train_loader))` grabs one batch for inspection.
- If the row count isn't divisible by the batch size, the last batch is smaller. Ours divides evenly.

### Shuffling is sampling without replacement

Each epoch, the DataLoader shuffles the row indices 0 to 16,511 and deals them out in chunks. **Every row is used exactly once per epoch**, with no repeats and no skips, like dealing a shuffled deck of cards. Over 50 epochs, each row is seen 50 times in different random positions. Sampling *with* replacement is possible with a custom sampler, but it isn't the default.

### Key vocabulary

| Term | Meaning |
|---|---|
| **Batch** | A small group of rows processed together (64 here) |
| **Iteration / step** | One batch processed and one weight update |
| **Epoch** | One full pass through all training data (258 steps here) |

### Why mini-batches?

- **More updates:** 258 weight updates per epoch instead of 1, so learning is much faster.
- **Helpful noise:** each batch is a slightly different sample, and the noise helps the model avoid getting stuck.
- **Memory:** huge datasets don't fit in memory at once.

---

## 9. Neural Network Fundamentals

### A single neuron is a weighted sum plus a bias

```
output = w1·x1 + w2·x2 + … + w8·x8 + b
```

- **Weights (w)** say how much each input matters, and in which direction.
- **Bias (b)** is a learnable offset.
- Training means **searching for the weights and biases that make predictions match reality.**
- A single neuron on its own is just **linear regression**.

### A layer is many neurons side by side

`nn.Linear(8, 32)` is 32 neurons, each looking at all 8 inputs. It turns 8 numbers into 32. Each output acts like a **learned feature detector** (for example, "wealthy coastal block"). Training discovers these detectors automatically.

Mathematically, a layer is one matrix multiplication:

```
output = input @ W + b
# input (batch, 8) @ W (8, 32) + b (32,) → output (batch, 32)
```

### Activations: why ReLU matters

**Stacking linear layers without activations is pointless.** A straight line fed into a straight line is still a straight line, so ten stacked linear layers collapse into the equivalent of one.

**ReLU** (Rectified Linear Unit): `ReLU(z) = max(0, z)`. Negatives become 0, and positives pass through unchanged.

That one kink lets the network bend and combine straight pieces into curves, capturing non-linear patterns like "price rises with income, but much faster near the coast."

**The recipe:** Linear → ReLU → Linear → ReLU → … → Linear

### The three shape rules

1. **The first layer's input = number of features.** We have 8 features, so `Linear(8, …)`.
2. **Each layer's output = the next layer's input.** `Linear(8, 32)` must be followed by `Linear(32, …)`. A mismatch raises a shape error.
3. **The last layer matches the task:**
   - **Regression** (one number): `Linear(…, 1)` with **no activation**, since prices can be any value and ReLU would chop off negatives.
   - **Classification** (N categories): `Linear(…, N)`, for example 10 outputs for digits 0 to 9.

### Choosing the hidden sizes

The middle sizes are **design choices (hyperparameters)**:

- Start small. 1 to 3 hidden layers of 32 to 128 neurons is plenty for tabular data.
- A funnel (64 → 32 → 1) is common, but expanding first (32 → 64 → 1) also works.
- **Too small → underfitting:** the model can't capture the patterns, so both training and test errors are high.
- **Too large → overfitting:** the model memorizes the training data, so training error is low but test error is high.

### Counting parameters

Each Linear layer has `inputs × outputs` weights plus `outputs` biases.

| Layer | Calculation | Parameters |
|---|---|---|
| Linear(8, 32) | 8×32 + 32 | 288 |
| Linear(32, 64) | 32×64 + 64 | 2,112 |
| Linear(64, 1) | 64×1 + 1 | 65 |
| **Total** | | **2,465** |

```python
sum(p.numel() for p in model.parameters())   # 2465
```

---

## 10. Weight Initialization

### Why not start with all zeros?

If every weight starts the same, every neuron computes the same thing and receives the same update, so they stay identical forever. Random initialization **breaks the symmetry**, letting each neuron learn something different.

### Why the `* (2 / inputs) ** 0.5` scaling?

```python
W1 = (torch.randn(8, 32) * (2 / 8) ** 0.5).requires_grad_()
```

- `torch.randn` draws numbers from a standard normal distribution (mean 0, std 1).
- Each neuron **sums** many inputs × weights. With more inputs, the sum grows larger. Across several layers, values can **explode** (grow huge) or **vanish** (shrink to near zero), and then training breaks or stalls.
- Multiplying by `sqrt(2 / fan_in)`, where `fan_in` is the number of inputs to the layer, keeps the output size roughly stable from layer to layer.
- The **2** compensates for ReLU zeroing out about half the values.
- This is called **He initialization** (or **Kaiming initialization**), and it's designed for ReLU networks.
- `nn.Linear` applies a similar fan-in-based initialization automatically, which is why you don't do this manually with built-in layers.

**Biases start at zero.** That's fine, because the random weights already break the symmetry.

### `requires_grad`

`requires_grad=True` tells PyTorch "this is a learnable value, so track how it affects the loss." Without it, `.grad` stays `None` and the value never learns.

`(tensor * scale).requires_grad_()` sets the flag **after** scaling, so the weight itself is the tracked leaf tensor rather than an intermediate result.

### Reproducibility

`torch.manual_seed(0)` makes all random numbers (initial weights, shuffles) the same on every run, so experiments can be compared fairly.

---

## 11. Loss Functions: MSE and RMSE

### Mean squared error (MSE): the training loss

```
MSE = mean( (prediction − actual)² )
```

```python
loss = ((preds - y_batch) ** 2).mean()   # by hand
loss = nn.MSELoss()(preds, y_batch)      # built-in, identical
```

1. `preds - y_batch` gives each miss.
2. `** 2` squares it, so negative and positive misses both count, and **big misses are punished more**.
3. `.mean()` averages them into one number.

**Loss** and **cost function** mean the same thing: a single number for how wrong the model is. Training tries to minimize it.

### Root mean squared error (RMSE): the human-readable error

```
RMSE = sqrt(MSE)
```

MSE is in squared units ("squared $100,000s"), which is meaningless to humans. Taking the square root returns to the original units, and multiplying by 100,000 converts to dollars. **RMSE is roughly "the typical size of a miss."**

Example: MSE 0.2956 → RMSE 0.544 → **about $54,000**.

### Why MSE for regression?

It's smooth and differentiable everywhere, so gradients are well-behaved. It also penalizes big errors heavily. (MAE, the mean absolute error, is an alternative that's less sensitive to outliers.)

---

## 12. The Training Loop

### The five steps for every batch

| # | Step | Raw code | Built-in |
|---|---|---|---|
| 1 | **Predict** (forward pass) | `preds = forward(x_batch)` | `preds = model(x_batch)` |
| 2 | **Measure** (loss) | `((preds - y_batch)**2).mean()` | `loss_fn(preds, y_batch)` |
| 3 | **Clear** old gradients | `p.grad.zero_()` | `optimizer.zero_grad()` |
| 4 | **Backpropagate** | `loss.backward()` | `loss.backward()` |
| 5 | **Update** weights | `p -= lr * p.grad` | `optimizer.step()` |

### What each step really does

**Forward pass:** data flows through the layers to produce predictions. PyTorch records every operation as a **computation graph**.

**`loss.backward()` (backpropagation):** walks the computation graph backwards using the chain rule from calculus. It calculates the **gradient** of the loss with respect to every parameter, meaning "if I increase this weight slightly, how much does the loss change, and in which direction?" The results land in each parameter's `.grad`. This is called **autograd** (automatic differentiation).

**Gradient descent update:** `p = p − lr × gradient`. The gradient points uphill, toward more error, so we step the **opposite** way, downhill toward less error.

**Zeroing gradients:** PyTorch **accumulates** gradients, adding new ones to old ones by default. Without clearing them each batch, gradients pile up and training breaks.

**`torch.no_grad()`:** the weight update itself must not be tracked as part of the model's computation. Updating parameters in place inside a tracked context also raises an error. We use the same context for evaluation, since we're just looking and tracking would waste time and memory.

### Learning rate (lr)

The size of each step downhill.

- **Too high:** the model overshoots, bounces around, and the loss may become `nan` (not a number).
- **Too low:** progress is painfully slow.
- **Typical values:** 0.01 for plain SGD, 0.001 for Adam.

### Optimizers

| Optimizer | How it updates |
|---|---|
| **SGD** (what our raw loop does) | `p -= lr * grad`, the same step size rule for every weight |
| **Adam** | Keeps a running average of past gradients (**momentum**, to smooth noisy steps) and of past squared gradients (to give **each weight its own adaptive step size**). Usually converges faster with less tuning. |

### `.item()`

Converts a one-number tensor into a plain Python number. Use it when accumulating the loss for printing, so you don't keep the computation graph alive in memory.

### Average epoch loss

Sum `loss.item()` over all batches, then divide by the number of batches. That gives one number per epoch to watch fall.

### `model.train()` and `model.eval()`

These switch layers like Dropout and BatchNorm between training and evaluation behavior. Our model has neither, so they don't change anything here, but always using them is a good habit.

---

## 13. Evaluation

```python
with torch.no_grad():
    test_preds = forward(x_test)                      # all 4,128 rows at once
    test_rmse  = ((test_preds - y_test) ** 2).mean().sqrt() * 100000

    baseline   = y_train.mean()                       # ≈ 2.07 → $207,000
    base_rmse  = ((baseline - y_test) ** 2).mean().sqrt() * 100000

print(f"Model test error:    ${test_rmse:,.0f}")
print(f"Baseline test error: ${base_rmse:,.0f}")
```

- There's no batching at test time, because no weights are being updated.
- `f"${value:,.0f}"` formats dollars: `,` adds thousands separators, and `.0f` means no decimals.
- The single baseline number is **broadcast** against all 4,128 actual prices automatically.

### The three questions evaluation answers

1. **How wrong is it on new data?** Test RMSE: $56,109.
2. **Is it better than doing nothing smart?** Compare to the baseline that always guesses the training average: $114,486. Beating the baseline proves the features carry real information.
3. **Did it learn or memorize?** Compare training error with test error:

| Pattern | Diagnosis | Fixes |
|---|---|---|
| Train ≈ test, both low | Good fit, generalizes | Done |
| Train low, test much higher | **Overfitting** (memorized) | Smaller model, fewer epochs, more data, regularization |
| Both high | **Underfitting** (too simple) | Bigger model, more epochs, better features |

**Also inspect a few individual predictions.** Averages can hide nonsense, so a quick look by eye is a sanity check. Our capped $500,000 block was predicted at $407,900, which is reasonable given that its true value may be higher.

---

## 14. Hyperparameters and Experiments

**Hyperparameters** are settings you choose (they aren't learned): layer sizes, learning rate, epochs, batch size, optimizer, activation.

**Parameters** are what the model learns: weights and biases.

The `run_experiment()` function retrains from scratch with chosen knobs and the same seed, which makes comparisons fair. Guess the result **before** each run to build intuition.

| Experiment | What it teaches |
|---|---|
| `run_experiment()` | Control: should reproduce $54,014 / $56,109 |
| `use_relu=False` | Without activations, the network collapses to a straight-line model |
| `epochs=100` | Whether more training helps, and whether the train/test gap grows |
| `use_adam=True, lr=0.001` | Adaptive optimizer vs plain SGD |
| `lr=0.1` / `lr=0.001` | Learning rate too high (unstable, maybe `nan`) vs too low (slow) |
| `h1=4, h2=4` / `h1=256, h2=256` | Underfitting vs overfitting risk |

**Change one thing at a time**, or you can't tell which change caused the result.

---

## 15. Raw Code vs Built-in PyTorch

We built everything by hand first to understand what the built-in tools do.

| Concept | Raw version | Built-in version |
|---|---|---|
| Layer | `x @ W + b` | `nn.Linear(in, out)` |
| ReLU | `torch.clamp(z, min=0)` | `nn.ReLU()` |
| Stacking layers | A `forward()` function | `nn.Sequential(...)` |
| Initialization | `randn * sqrt(2/fan_in)` | Automatic in `nn.Linear` |
| Loss | `((p - y)**2).mean()` | `nn.MSELoss()` |
| Update | `p -= lr * p.grad` | `optim.SGD` / `optim.Adam` + `.step()` |
| Clear gradients | `p.grad.zero_()` | `optimizer.zero_grad()` |
| Shuffle + batch | `torch.randperm` + slicing | `DataLoader(shuffle=True)` |
| Collect parameters | `params = [W1, b1, ...]` | `model.parameters()` |

### nn.Sequential vs a custom nn.Module

**`nn.Sequential`** works when data flows straight through, layer after layer:

```python
model = nn.Sequential(
    nn.Linear(8, 32), nn.ReLU(),
    nn.Linear(32, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
```

**A custom `nn.Module`** is needed for anything non-sequential, like multiple inputs or skip connections. You define the layers in `__init__` and the data flow in `forward`:

```python
class HousePriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 32)
        self.layer2 = nn.Linear(32, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.out(x)

model = HousePriceNet()
```

`model(x)` calls `forward(x)` for you. Don't call `forward` directly.

---

## 16. Function Reference

### scikit-learn

| Function | Purpose |
|---|---|
| `fetch_california_housing()` | Downloads and loads the dataset |
| `train_test_split(X, y, test_size, random_state)` | Splits into train and test sets |
| `StandardScaler()` | Creates a scaler |
| `.fit_transform(x)` / `.transform(x)` | Learn and scale / scale with stored values |

### PyTorch: tensors

| Function | Purpose |
|---|---|
| `torch.tensor(data, dtype=torch.float32)` | Create a tensor |
| `.shape` | Dimensions |
| `.view(-1, 1)` / `.reshape(-1, 1)` | Change shape |
| `torch.randn(a, b)` | Random normal tensor |
| `torch.zeros(n)` | Tensor of zeros |
| `torch.randperm(n)` | Shuffled indices 0 to n−1 |
| `torch.clamp(z, min=0)` | Clip values (ReLU by hand) |
| `@` | Matrix multiplication |
| `.mean()`, `.sqrt()` | Average, square root |
| `.item()` | One-element tensor to Python number |
| `.numel()` | Number of elements |
| `torch.manual_seed(n)` | Reproducible randomness |

### PyTorch: autograd

| Function | Purpose |
|---|---|
| `requires_grad=True` / `.requires_grad_()` | Track gradients for this tensor |
| `loss.backward()` | Compute all gradients (backpropagation) |
| `.grad` | Where a tensor's gradient is stored |
| `.grad.zero_()` | Clear a gradient |
| `with torch.no_grad():` | Turn off tracking (updates and evaluation) |

### PyTorch: nn, optim, data

| Function | Purpose |
|---|---|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()` | ReLU activation |
| `nn.Sequential(...)` | Stack layers in order |
| `nn.Module` | Base class for custom models |
| `nn.MSELoss()` | Mean squared error loss |
| `model.parameters()` | All learnable weights |
| `model.train()` / `model.eval()` | Training / evaluation mode |
| `optim.SGD(params, lr)` / `optim.Adam(params, lr)` | Optimizers |
| `optimizer.zero_grad()` / `optimizer.step()` | Clear gradients / update weights |
| `TensorDataset(x, y)` | Pair features with targets |
| `DataLoader(ds, batch_size, shuffle)` | Batch and shuffle |
| `next(iter(loader))` | Grab one batch |

---

## 17. Bugs We Hit and Lessons Learned

| Bug | Cause | Lesson |
|---|---|---|
| `pip install torch` → "from versions: none" | No PyTorch build for that Python version and architecture | Use a supported Python (3.12) with a native architecture |
| `y_train = torch.tensor(x_train, ...)` | Copy-paste typo: y was built from x | Check shapes after every step. y must be `(N, 1)`, not `(N, 8)` |
| "To copy construct from a tensor…" warning | Ran the conversion cell twice | Notebooks keep state, so rerun from the split cell or Restart & Run All |
| y shape `(N,)` vs predictions `(N, 1)` | Missing reshape | Silent broadcasting gives wrong losses. Always `.view(-1, 1)` |
| `NameError: train_loader` | Cell never run | Run cells in order |

**General habits:**

- Print shapes after every transformation.
- Test the model on a random input (`torch.randn(1, 8)`) before training.
- Take a "before training" snapshot to compare with afterwards.
- Compare against a baseline.
- Fix random seeds when comparing experiments.

---

## 18. Interview Questions

**Q: What type of neural network did you build?**
A Multilayer Perceptron (MLP), also called a feedforward, fully connected neural network. It has 8 inputs, two hidden layers (32 and 64 neurons with ReLU), and a single linear output neuron for regression. I trained it with supervised learning to predict house prices from tabular data.

**Q: What does "feedforward" mean?**
Data moves in one direction, from input to output, with no loops or memory between examples. That's unlike RNNs, which carry state from one step to the next.

**Q: What's the difference between a parameter and a hyperparameter?**
Parameters (weights, biases) are learned during training. Hyperparameters (learning rate, layer sizes, epochs, batch size) are chosen by you before training.

**Q: Why do we need activation functions?**
Without them, stacked linear layers collapse into a single linear function, so the network could only learn straight-line relationships. Non-linear activations like ReLU let it model curves and interactions.

**Q: Why ReLU over sigmoid or tanh?**
It's cheap to compute, and it doesn't saturate for positive values, so gradients don't vanish in deep networks. Its downside is "dying ReLU": neurons stuck outputting 0.

**Q: Why scale features?**
Features on very different scales make gradients unbalanced, and no single learning rate works for all weights. Scaling speeds up and stabilizes training.

**Q: What is data leakage? Give an example.**
When information from the test set influences training. Example: fitting a StandardScaler on the full dataset before splitting.

**Q: Explain backpropagation simply.**
After computing the loss, we apply the chain rule backwards through the network to find how much each weight contributed to the error (its gradient). Then we adjust each weight in the opposite direction of its gradient.

**Q: Why call `optimizer.zero_grad()`?**
PyTorch accumulates gradients by default. Without zeroing, each batch's gradients add onto the previous ones and the updates become wrong.

**Q: What are batch, iteration, and epoch?**
A batch is a subset of rows processed together. An iteration is one batch plus one weight update. An epoch is one full pass over the training data.

**Q: Why mini-batches rather than full-batch or single-example training?**
They balance stable gradient estimates against frequent updates, fit in memory, and add helpful noise.

**Q: What happens if the learning rate is too high or too low?**
Too high: the loss oscillates or diverges, possibly to `nan`. Too low: training is very slow and may stall.

**Q: How do you detect overfitting?**
Training error keeps falling while test (or validation) error stays flat or rises, creating a large gap between them.

**Q: How do you fix overfitting?**
More data, a smaller model, fewer epochs (early stopping), regularization (weight decay, dropout), or data augmentation.

**Q: Why not initialize weights to zero?**
All neurons would compute the same output and receive the same gradient, so they'd never differentiate (the symmetry problem).

**Q: What is He initialization?**
Random weights scaled by `sqrt(2 / fan_in)`. It keeps activation sizes stable across ReLU layers, preventing exploding or vanishing values.

**Q: MSE vs RMSE?**
MSE is the training loss: smooth, differentiable, and punishing big errors. RMSE is the square root of MSE, back in the target's units, so it's easier to interpret.

**Q: What does `torch.no_grad()` do and when is it used?**
It disables gradient tracking. It's used for evaluation and inference (faster, less memory) and for manual weight updates.

**Q: SGD vs Adam?**
SGD uses one fixed step rule for all weights. Adam adds momentum and per-weight adaptive learning rates, which usually means faster convergence with less tuning.

**Q: Why use a baseline?**
A raw error number has no context. A simple baseline (always predicting the mean) shows whether the model learned anything meaningful.

**Q: What does the final layer look like for regression vs classification?**
Regression: one output with no activation. Classification: one output per class, with logits passed to cross-entropy loss.

**Q: Why does the target need shape `(N, 1)` rather than `(N,)`?**
To match the model's output shape. Otherwise broadcasting silently computes the loss over an N × N grid.

---

## 19. Write It Yourself: Full Code Templates

### Version A: from scratch (raw tensors)

```python
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Data
housing = fetch_california_housing()
X, y = housing.data, housing.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale (fit on train only)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 3. Tensors (y as a column)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 4. Weights (He initialization)
torch.manual_seed(0)
W1 = (torch.randn(8, 32) * (2 / 8) ** 0.5).requires_grad_()
b1 = torch.zeros(32, requires_grad=True)
W2 = (torch.randn(32, 64) * (2 / 32) ** 0.5).requires_grad_()
b2 = torch.zeros(64, requires_grad=True)
W3 = (torch.randn(64, 1) * (2 / 64) ** 0.5).requires_grad_()
b3 = torch.zeros(1, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]

# 5. Forward pass
def forward(x):
    h1 = torch.clamp(x @ W1 + b1, min=0)   # Linear + ReLU
    h2 = torch.clamp(h1 @ W2 + b2, min=0)  # Linear + ReLU
    return h2 @ W3 + b3                    # Linear, no activation

# 6. Training loop (manual batching + SGD)
lr, batch_size, n = 0.01, 64, x_train.shape[0]
for epoch in range(50):
    perm = torch.randperm(n)
    total_loss, num_batches = 0, 0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        x_batch, y_batch = x_train[idx], y_train[idx]

        preds = forward(x_batch)                     # predict
        loss = ((preds - y_batch) ** 2).mean()       # MSE
        loss.backward()                              # gradients

        with torch.no_grad():
            for p in params:
                p -= lr * p.grad                     # update
                p.grad.zero_()                       # clear

        total_loss += loss.item()
        num_batches += 1
    print(f"Epoch {epoch + 1}: loss = {total_loss / num_batches:.4f}")

# 7. Evaluate
with torch.no_grad():
    test_rmse = ((forward(x_test) - y_test) ** 2).mean().sqrt() * 100000
    base_rmse = ((y_train.mean() - y_test) ** 2).mean().sqrt() * 100000
print(f"Model test error:    ${test_rmse:,.0f}")
print(f"Baseline test error: ${base_rmse:,.0f}")
```

### Version B: standard PyTorch style

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1–3. Data, split, scale, tensors (same as Version A)
housing = fetch_california_housing()
X, y = housing.data, housing.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 4. DataLoader
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

# 5. Model, loss, optimizer
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(8, 32), nn.ReLU(),
    nn.Linear(32, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
for epoch in range(50):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: loss = {total_loss / len(train_loader):.4f}")

# 7. Evaluate
model.eval()
with torch.no_grad():
    test_rmse = loss_fn(model(x_test), y_test).sqrt().item() * 100000
print(f"Model test error: ${test_rmse:,.0f}")

# 8. Save and reload
torch.save(model.state_dict(), "house_model.pt")
# model.load_state_dict(torch.load("house_model.pt"))
```

### Self-test checklist

Try writing Version B from memory, then check:

- [ ] Split before scaling, and fit the scaler on train only
- [ ] float32 tensors, with y reshaped to `(N, 1)`
- [ ] First layer input = number of features, and layer sizes chain correctly
- [ ] No activation on the final regression layer
- [ ] The five loop steps in order: predict → loss → zero_grad → backward → step
- [ ] Evaluation under `torch.no_grad()` and `model.eval()`
- [ ] Test RMSE compared against a baseline and against training RMSE