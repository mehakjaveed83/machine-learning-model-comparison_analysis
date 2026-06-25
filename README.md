# Classifying Non-Linear Decision Boundaries: Logistic Regression vs. Neural Networks

A systematic empirical study comparing the performance of **Logistic Regression** and **Multilayer Perceptron (MLP)** classifiers on a 2D binary classification problem defined by a **quadratic decision boundary** — where linear models are structurally limited and neural networks are expected to shine.

---

## Overview

This notebook investigates how well classical and neural classifiers handle a fundamentally non-linear separation task. Points in 2D space are labelled based on whether they fall above or below the curve `y = ax² + x`, and both models are stress-tested across three controlled axes of variation:

- The **curvature parameter `a`** of the decision boundary
- The **size of the training dataset**
- The **class balance ratio**

A final experiment explores the effect of **MLP architecture depth and width** on classification accuracy.

---

## Experiments

### 1. Baseline: Dataset Visualisation

A balanced dataset of 1,000 points is generated with `a = 1`. Points are scattered uniformly in `[-10, 10]²` and labelled according to the quadratic boundary. The dataset is visualised with the true decision boundary overlaid.

> Class balancing is achieved by downsampling the majority class, ensuring equal class representation regardless of the boundary's geometry.

### 2. Effect of Boundary Curvature (Parameter `a`)

Models are evaluated across `a ∈ {-0.2, -0.1, 0.1, 0.2, 1.0}`.

| Parameter `a` | Boundary Shape | Expected Linear Model Difficulty |
|:-:|:-:|:-:|
| ±0.1 | Near-linear | Low |
| ±0.2 | Mild curvature | Moderate |
| 1.0 | Strong parabola | High |

As `|a|` increases, the boundary becomes more curved and logistic regression — which can only learn a linear separator in the original feature space — struggles increasingly. The MLP, by contrast, learns non-linear mappings through its hidden layers.

### 3. Effect of Dataset Size

With `a = 1` fixed, both models are evaluated on datasets of size `{100, 500, 900, 1300}`.

This experiment reveals the **sample efficiency** of each model: whether the MLP's additional capacity pays off with small data, or whether it requires more examples to converge reliably.

### 4. Effect of Class Imbalance

Class balance (proportion of class 1) is varied across `{0.1, 0.3, 0.5, 0.7, 0.9}` to examine how skewed distributions affect each classifier.

Logistic regression can become biased toward the majority class under imbalance. The MLP is also susceptible, though its decision surface flexibility can sometimes compensate.

### 5. MLP Architecture Search

With balanced data and `a = 1`, the following architectures are compared:

| Architecture | Hidden Layers | Total Neurons |
|:-:|:-:|:-:|
| `(10,)` | 1 | 10 |
| `(10, 10)` | 2 | 20 |
| `(10, 10, 10)` | 3 | 30 |
| `(20, 10)` | 2 | 30 |
| `(10, 20)` | 2 | 30 |

This isolates the impact of **depth vs. width** on learning a quadratic boundary.

---

## Technical Stack

| Library | Purpose |
|:-:|:-:|
| `numpy` | Data generation, boundary computation, array ops |
| `matplotlib` | Visualisation of datasets and results |
| `scikit-learn` | Model training, evaluation, train/test splitting |

**Models used:**
- `LogisticRegression` — L2-regularised, solver defaults
- `MLPClassifier` — ReLU activations, Adam optimiser, up to 2000 iterations

---

## Key Insights

- **Logistic regression hits a ceiling** as boundary curvature increases — its linear decision surface simply cannot represent a parabola without feature engineering.
- **The MLP generalises well** even at moderate dataset sizes (≥ 500 samples), demonstrating that its inductive bias is better suited to this geometry.
- **Class imbalance degrades both models** but the MLP recovers more gracefully at moderate imbalance ratios.
- **Architecture depth matters more than width** for this task: adding a second hidden layer consistently outperforms simply making a single layer wider.

---

## Repository Structure

```
├── AI_question_number_two.ipynb   # Full experimental notebook
└── README.md                      # This file
```

---

## Running the Notebook

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install numpy matplotlib scikit-learn

# Launch the notebook
jupyter notebook AI_question_number_two.ipynb
```

All experiments use `random_state=42` for full reproducibility.

---

## Concepts Demonstrated

`Binary Classification` · `Non-Linear Decision Boundaries` · `Logistic Regression` · `Multilayer Perceptron` · `Class Balancing` · `Hyperparameter Sensitivity` · `Model Comparison` · `scikit-learn`
