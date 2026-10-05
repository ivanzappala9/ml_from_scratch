# Machine Learning from Scratch

An educational implementation of classification algorithms and preprocessing tools in Python, originally developed for a university assignment and later reorganized into separate modules.

The goal is to understand and implement the algorithms directly, without relying on ready-made classifiers.

## Dependencies

- **NumPy:** numerical operations and algorithm implementation.
- **SciPy:** numerically stable sigmoid function (`expit`).
- **Matplotlib:** learning curves.
- **scikit-learn:** dataset loading only.

## Project Structure

| File | Contents |
|---|---|
| `src/preprocessing.py` | StandardNormalizer and OneHotEncoder |
| `src/model_selection.py` | Stratified train/test split |
| `src/metrics.py` | Accuracy and binary cross-entropy |
| `src/logistic_regression.py` | Binary logistic regression with mini-batch training |
| `src/multiclass.py` | One vs Rest classification |
| `src/naive_bayes.py` | Categorical Naive Bayes |
| `src/experiment.py` | Dataset experiments |
| `src/test_*.py` | Component verification scripts |

Features are represented by a two-dimensional array `X`, with samples as rows and features as columns. Labels are stored in a one-dimensional array `y`.

Classifiers expose `fit`, `predict`, and `predict_proba`. Probability columns follow the order of labels in `model.classes`.

## Preprocessing and Metrics

**StandardNormalizer** estimates feature means and standard deviations from the training set:

$$
z_j = \frac{x_j-\mu_j}{\sigma_j}.
$$

For constant features, the scaling factor is set to 1 to avoid division by zero.

**OneHotEncoder** learns categories during `fit` and preserves their mapping during `transform`. Unknown categories raise an error.

The **stratified split** approximately preserves class proportions and supports reproducible sampling through `random_state`.

Implemented metrics:

- **Accuracy:** fraction of correctly classified samples.
- **Binary cross-entropy:** average loss on predicted positive-class probabilities:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

Probabilities are clipped before evaluating logarithms to avoid numerical issues at 0 and 1.

## Classifiers

### Binary Logistic Regression

The model estimates the probability of class 1 using a sigmoid:

$$
P(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^{T}\mathbf{x}+b),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Parameters are learned by minimizing binary cross-entropy using mini-batch gradient descent. Each epoch processes all training samples in shuffled order.

The implementation includes:

- An intercept represented by an additional constant feature.
- Gradients averaged over each mini-batch.
- Optional learning-rate decay.
- Training and validation loss histories.
- Early stopping with restoration of the best validation weights.

Labels must be encoded as 0 and 1. Predictions use a probability threshold of 0.5.

### One vs Rest Logistic Regression

One binary classifier is trained for each class, distinguishing that class from all remaining classes.

Prediction selects the class with the highest score. `predict_proba` divides scores by their row sum to produce a distribution over classes. This preserves the predicted class but does not guarantee calibrated probabilities.

### Categorical Naive Bayes

The classifier applies Bayes’ theorem under the assumption that features are conditionally independent given the class:

$$
P(c\mid\mathbf{x})\propto P(c)\prod_j P(x_j\mid c).
$$

Class priors are estimated from class frequencies. Conditional probabilities use additive smoothing:

$$
P(x_j=a\mid c)=
\frac{N_{c,j,a}+\alpha}{N_c+\alpha K_j},
$$

where $K_j$ is the number of categories observed for feature $j$ in the training set. Setting $\alpha=1$ gives Laplace smoothing.

Predictions are computed in log space and normalized using a numerically stable calculation. Categories never observed during training are rejected.

## Verification

Verification scripts use simple examples and assertions to check:

- Output shapes and probability normalization.
- Predictions on separable synthetic problems.
- Replacement of previous classifiers when refitting One vs Rest.
- Early stopping and restoration of the best weights.
- Naive Bayes probabilities against manual calculations.
- Numerical stability with many categorical features.
- Rejection of unknown categories.

These are targeted checks rather than an exhaustive test suite.

## Experiments

The experiment script covers:

1. **Breast Cancer:** binary logistic regression with standardization and early stopping.
2. **Iris:** One vs Rest logistic regression with standardization.
3. **Discretized Iris:** Categorical Naive Bayes with three bins per feature, defined using training-set quantiles.

Preprocessing parameters are estimated exclusively from the training set. Evaluation includes accuracy, a training-majority-class baseline, and a confusion matrix.

The initial Breast Cancer validation run produced:

| Metric | Value |
|---|---:|
| Logistic regression accuracy | 0.9737 |
| Majority-class baseline accuracy | 0.6316 |
| Binary cross-entropy | 0.1166 |
| Completed epochs | 47 |

These results refer to a single validation split, not the final test set.

## Running the Project

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib scikit-learn
```

Run the experiments:

```bash
python src/experiment.py
```

Run an individual verification script:

```bash
python src/test_naive_bayes.py
```

## Limitations

This project is intended for educational use. It does not implement weight regularization, probability calibration, or automatic missing-value handling.

Performance depends on the dataset and split. The Iris experiments use different feature representations for logistic regression and Naive Bayes, so they are not a comparison under identical inputs.
