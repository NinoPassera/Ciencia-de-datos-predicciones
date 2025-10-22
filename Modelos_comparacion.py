# %% Imports y configuraciones
import os, time, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedShuffleSplit,
    cross_validate, RandomizedSearchCV, learning_curve
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, top_k_accuracy_score, make_scorer
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# %% Carga de datos y definición de features
DATA_FILE = "dataset_modelo_final.csv"   #
RANDOM_SEED = 42
CV_FOLDS = 5

assert os.path.exists(DATA_FILE), f"No se encontró {DATA_FILE}"

df = pd.read_csv(DATA_FILE)
print(f"[{datetime.now()}] Dataset cargado: {len(df):,} filas")

features_originales = [
    'origen_lat', 'origen_lon',
    'hora_salida', 'dia_semana', 'mes',
    'viajes_totales', 'semanas_activas', 'viajes_por_semana', 'duracion_promedio_min'
]
features_mejoradas = [
    'periodo_dia_numerico','es_fin_semana','es_hora_pico','zona_origen',
    'capacidad_origen','estaciones_cercanas_origen','variedad_destinos','variedad_origenes',
    'consistencia_horaria','distancia_promedio_usuario','dia_favorito',
    'frecuencia_lunes','frecuencia_martes','frecuencia_miercoles',
    'frecuencia_jueves','frecuencia_viernes','frecuencia_sabado','frecuencia_domingo'
]
features = features_originales + features_mejoradas

X = df[features].fillna(0).astype(np.float32)
y = df['destino'].astype(str)

print(f"Features totales: {len(features)} | Clases: {y.nunique()}")

# %% (Opcional) Limitar filas con muestra estratificada para acelerar
MAX_ROWS = 50000   # ejemplo: 80000 ó 120000; None = usar todo

if MAX_ROWS is not None and MAX_ROWS < len(X):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_ROWS, random_state=RANDOM_SEED)
    idx, _ = next(sss.split(X, y))
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)
    print(f"[MUESTRA] Usando {len(X):,} filas estratificadas")

    # %% Partición Train/Test estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# %% Definición de modelos con Pipelines
# Nota: LR y SVC requieren StandardScaler; RF no.
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs", penalty="l2", C=1.0,
            max_iter=8000, class_weight="balanced", random_state=RANDOM_SEED
        ))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=30,
            min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", bootstrap=True,
            random_state=RANDOM_SEED, n_jobs=-1
        ))
    ]),
    "SVC_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=RANDOM_SEED
        ))
    ])
}

# Scorers para cross_validate
scorers = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "top5": make_scorer(top_k_accuracy_score, needs_proba=True, k=5)
}


# %% Comparación de modelos (CV en Train) + evaluación en Test
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_results = {}
test_results = {}

for name, pipe in models.items():
    print(f"\n=== {name} ===")
    # CV en train
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scorers, n_jobs=-1, return_train_score=False)
    cv_results[name] = {metric: (cv_res[f"test_{metric}"].mean(), cv_res[f"test_{metric}"].std())
                        for metric in scorers.keys()}
    print("CV (mean ± 2*std):")
    for metric, (m, s) in cv_results[name].items():
        print(f"  {metric:>8}: {m:.4f} ± {2*s:.4f}")

    # Fit completo en train y evaluación en test
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    # Top-5 (si el modelo soporta predict_proba)
    try:
        proba = pipe.predict_proba(X_test)
        top5 = top_k_accuracy_score(y_test, proba, k=5, labels=getattr(pipe, "classes_", None))
    except Exception:
        top5 = np.nan

    test_results[name] = {"accuracy": acc, "f1_macro": f1m, "top5": top5, "fit_time_s": fit_time}
    print(f"Test -> accuracy: {acc:.4f} | f1_macro: {f1m:.4f} | top5: {top5:.4f} | fit_time: {fit_time:.2f}s")

# %% Selección del mejor modelo por accuracy CV (podés cambiar a F1 macro si te interesa más)
best_name = max(cv_results, key=lambda k: cv_results[k]["accuracy"][0])
print(f"\nMejor por CV (accuracy): {best_name} -> {cv_results[best_name]}")

best_pipe = models[best_name]

# %% Búsqueda de hiperparámetros (RandomizedSearchCV) sobre el mejor modelo
# Ajustá el espacio de búsqueda según el modelo ganador

param_distributions = {}
if best_name == "RandomForest":
    param_distributions = {
        "clf__n_estimators": [300, 600, 800],
        "clf__max_depth": [None, 16, 24, 32],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", 0.5, 0.3],
        "clf__class_weight": [None, "balanced_subsample"],
        "clf__bootstrap": [True]
    }
elif best_name == "LogisticRegression":
    param_distributions = {
        "clf__C": np.logspace(-2, 2, 15),
        "clf__solver": ["lbfgs", "saga"],
        "clf__penalty": ["l2", "elasticnet"],   # elasticnet requiere saga
        "clf__l1_ratio": [None, 0.1, 0.3, 0.5, 0.7]  # usado si elasticnet
    }
elif best_name == "SVC_RBF":
    param_distributions = {
        "clf__C": np.logspace(-1, 2, 10),
        "clf__gamma": ["scale", "auto", 0.1, 0.01, 0.001]
    }

N_ITER = 25
search = RandomizedSearchCV(
    estimator=best_pipe,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    verbose=1
)

print(f"\nRandomizedSearchCV sobre {best_name} (n_iter={N_ITER})…")
t0 = time.time()
search.fit(X_train, y_train)
dt = time.time() - t0
print(f"[OK] Búsqueda completada en {dt:.2f}s")
print("Mejores params:", search.best_params_)
print("CV best score (accuracy):", f"{search.best_score_:.4f}")

best_model = search.best_estimator_

# %% Evaluación final en Test + Reporte detallado + Confusion Matrix parcial
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

try:
    proba = best_model.predict_proba(X_test)
    top5 = top_k_accuracy_score(y_test, proba, k=5, labels=getattr(best_model, "classes_", None))
except Exception:
    top5 = np.nan

print("\n[FINAL EN TEST]")
print(f"Accuracy: {acc:.4f} | F1 macro: {f1m:.4f} | Top-5: {top5:.4f}\n")
print("[Classification report]")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix de las 15 clases más frecuentes (para que sea legible)
topN = 15
top_classes = y.value_counts().head(topN).index
mask = y_test.isin(top_classes)
cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_classes)

print(f"\nEtiquetas (Top {topN} por frecuencia):")
print(list(top_classes))
print("\nMatriz de confusión (subconjunto):")
print(cm)

# %% Diagnóstico de Overfitting / Underfitting
# Señales:
# - Overfitting: CV alto >> Test bajo, y/o gran gap Train vs CV en learning curve.
# - Underfitting: CV y Test ambos bajos y cercanos.

print("\n[Diagnóstico rápido]")
print("CV (mejor modelo, accuracy):", f"{search.best_score_:.4f}")
print("TEST (accuracy):", f"{acc:.4f}")
print("Gap (CV - Test):", f"{search.best_score_ - acc:.4f}")

# (Opcional) Learning curve para ver cómo evoluciona el score con más datos
plot_learning_curve = False
if plot_learning_curve:
    train_sizes, train_scores, valid_scores = learning_curve(
        best_model, X_train, y_train, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, shuffle=True, random_state=RANDOM_SEED
    )
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(train_sizes, valid_scores.mean(axis=1), marker="o", label="CV")
    plt.xlabel("Tamaño de entrenamiento"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Learning Curve")
    plt.show()

# %% Guardado del mejor modelo
ARTIFACT_PATH = f"mejor_modelo_{best_name}.pkl"
joblib.dump(best_model, ARTIFACT_PATH)
print(f"[GUARDADO] Modelo final guardado en: {ARTIFACT_PATH}")
