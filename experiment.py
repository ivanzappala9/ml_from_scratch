import numpy as np
from sklearn.datasets import load_breast_cancer
from model_selection import stratified_train_test_split
from preprocessing import StandardNormalizer
from logistic_regression import LogisticRegressionMiniBatch
from metrics import accuracy_score, crossed_entropy
from metrics import crossed_entropy as binary_cross_entropy


dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("Forma di X:", X.shape)
print("Forma di y:", y.shape)
print("Caratteristiche:", dataset.feature_names)
print("Nomi delle classi:", dataset.target_names)

classes, counts = np.unique(y, return_counts=True)

for label, count in zip(classes, counts):
    print(
        f"Classe {label} ({dataset.target_names[label]}): "
        f"{count} esempi"
    )

print("Valori tutti finiti:", np.isfinite(X).all())




X_dev, X_test, y_dev, y_test = stratified_train_test_split(
    X, y,
    test=0.20,
    random_state=42
)

X_train, X_val, y_train, y_val = stratified_train_test_split(
    X_dev, y_dev,
    test=0.25,
    random_state=43
)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


normalizer = StandardNormalizer()

X_train_scaled = normalizer.fit_transform(X_train)
X_val_scaled = normalizer.transform(X_val)
X_test_scaled = normalizer.transform(X_test)




model = LogisticRegressionMiniBatch(
    rate=0.1,
    rate_decay=False,
    n_epochs=1000,
    batch_tam=32,
    random_state=42,
    early_stopping=True,
    paciencia=20
)

model.fit(
    X_train_scaled,
    y_train,
    Xv=X_val_scaled,
    yv=y_val
)





val_pred = model.predict(X_val_scaled)
val_proba = model.predict_proba(X_val_scaled)[:, 1]

print("Epoche eseguite:", len(model.loss_history))
print("Accuracy validation:", accuracy_score(y_val, val_pred))
print(
    "Cross-entropy validation:",
    crossed_entropy(y_val, val_proba)
)

classes, counts = np.unique(y_train, return_counts=True)
majority_class = classes[np.argmax(counts)]
baseline_pred = np.full(y_val.shape, majority_class)

print(
    "Accuracy baseline:",
    accuracy_score(y_val, baseline_pred)
)


import matplotlib.pyplot as plt

epochs = np.arange(1, len(model.loss_history) + 1)

plt.plot(epochs, model.loss_history, label="Training")
plt.plot(epochs, model.val_loss_history, label="Validation")
plt.xlabel("Epoca")
plt.ylabel("Cross-entropy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



from sklearn.datasets import load_iris

from multiclass import LogisticRegressionOVR
from naive_bayes import NaiveBayes


def evaluate(name, model, X_test, y_test, y_train):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = model.classes

    assert y_pred.shape == y_test.shape
    assert proba.shape == (len(y_test), len(classes))
    assert np.isfinite(proba).all()
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)

    train_classes, counts = np.unique(y_train, return_counts=True)
    majority_class = train_classes[np.argmax(counts)]
    baseline_pred = np.full(y_test.shape, majority_class)

    confusion = np.zeros((len(classes), len(classes)), dtype=int)

    for i, true_class in enumerate(classes):
        for j, predicted_class in enumerate(classes):
            confusion[i, j] = np.sum(
                (y_test == true_class) & (y_pred == predicted_class)
            )

    print(f"\n{name}")
    print(f"Accuracy test: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Accuracy baseline: {accuracy_score(y_test, baseline_pred):.4f}")
    print("Ordine delle classi:", classes)
    print("Matrice di confusione (righe=vere, colonne=previste):")
    print(confusion)
    
    
    
    
evaluate(
    "Regressione logistica — Breast Cancer",
    model,
    X_test_scaled,
    y_test,
    y_train
)

print(
    "Cross-entropy test:",
    binary_cross_entropy(
        y_test,
        model.predict_proba(X_test_scaled)[:, 1]
    )
)



iris = load_iris()

Xi_train, Xi_test, yi_train, yi_test = stratified_train_test_split(
    iris.data,
    iris.target,
    test=0.20,
    random_state=42
)

iris_normalizer = StandardNormalizer()
Xi_train_scaled = iris_normalizer.fit_transform(Xi_train)
Xi_test_scaled = iris_normalizer.transform(Xi_test)

ovr = LogisticRegressionOVR(
    rate=0.1,
    rate_decay=False,
    n_epochs=300,
    batch_tam=16,
    random_state=42
)

ovr.fit(Xi_train_scaled, yi_train)

evaluate(
    "Regressione logistica One vs Rest — Iris",
    ovr,
    Xi_test_scaled,
    yi_test,
    yi_train
)




thresholds = np.quantile(Xi_train, [1 / 3, 2 / 3], axis=0)


def discretize(X, thresholds):
    result = np.empty(X.shape, dtype=int)

    for j in range(X.shape[1]):
        result[:, j] = np.digitize(X[:, j], thresholds[:, j])

    return result


Xi_train_cat = discretize(Xi_train, thresholds)
Xi_test_cat = discretize(Xi_test, thresholds)

for j in range(Xi_train_cat.shape[1]):
    assert np.isin(
        Xi_test_cat[:, j],
        np.unique(Xi_train_cat[:, j])
    ).all(), f"Categoria non osservata nel training, colonna {j}"

nb = NaiveBayes(alpha=1)
nb.fit(Xi_train_cat, yi_train)

evaluate(
    "Categorical Naive Bayes — Iris discretizzato",
    nb,
    Xi_test_cat,
    yi_test,
    yi_train
)
