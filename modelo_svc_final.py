#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support Vector Classifier (SVC) FINAL - Predicción de destinos
Usa todas las features, escalado, CV, Top-k y exporta artefactos
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from sklearn.inspection import permutation_importance

def main():
    print("=" * 70)
    print("SVC (RBF) FINAL - TODAS LAS CARACTERISTICAS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------- Controles de rendimiento --------------------
    MAX_ROWS    = 50000   # p. ej., 80000 para acelerar; None = usar todo
    RANDOM_SEED = 42
    CV_FOLDS    = 5
    PERM_REPEATS = 3     # repetición para permutation importance (bajar si está lento)
    # -----------------------------------------------------------------

    # Carga
    print("\nCargando dataset final...")
    try:
        df = pd.read_csv("dataset_modelo_final.csv")
        print(f"[OK] Dataset final cargado: {len(df):,} registros")
    except FileNotFoundError:
        print("[ERROR] No se encontró dataset_modelo_final.csv")
        print("Ejecuta primero: python crear_dataset_final.py")
        return

    # Features
    features_originales = [
        'origen_lat','origen_lon',
        'hora_salida','dia_semana','mes',
        'viajes_totales','semanas_activas','viajes_por_semana','duracion_promedio_min'
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

    print(f"Features originales: {len(features_originales)}")
    print(f"Features mejoradas: {len(features_mejoradas)}")
    print(f"Total features: {len(features)}")
    print(f"Destinos únicos: {y.nunique()}")

    # (Opcional) limitar filas con muestra estratificada
    if MAX_ROWS is not None and MAX_ROWS < len(X):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_ROWS, random_state=RANDOM_SEED)
        idx_sample, _ = next(sss.split(X, y))
        X = X.iloc[idx_sample].reset_index(drop=True)
        y = y.iloc[idx_sample].reset_index(drop=True)
        print(f"[MUESTRA] Usando muestra estratificada de {len(X):,} filas")

    # Split
    print("\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Entrenamiento: {len(X_train):,}")
    print(f"Prueba: {len(X_test):,}")

    # Pipeline SVC (con escalado)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,       # para predict_proba y top-k
            random_state=RANDOM_SEED
        ))
    ])

    # Entrenamiento
    print("\nEntrenando SVC (RBF) final...")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[OK] Entrenamiento completado en {train_time:.2f} segundos")

    # Evaluación en test
    print("\nEvaluando modelo final...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Top-k
    proba = pipe.predict_proba(X_test)
    acc_top3 = top_k_accuracy_score(y_test, proba, k=3, labels=pipe.classes_)
    acc_top5 = top_k_accuracy_score(y_test, proba, k=5, labels=pipe.classes_)

    print(f"\n[RESULTADOS FINALES]")
    print(f"Accuracy en test: {acc*100:.2f}%")
    print(f"Top-3 accuracy: {acc_top3*100:.2f}%")
    print(f"Top-5 accuracy: {acc_top5*100:.2f}%")
    print("\n[Classification report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Validación cruzada (en train para no tocar test)
    print("\nRealizando validación cruzada...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy promedio CV: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

    # Importancia por permutación (costosa). La hacemos sobre un subconjunto del test para acelerar.
    print("\nCalculando Permutation Importance (subconjunto del test)...")
    subset = min(5000, len(X_test))  # limita para rendimiento
    Xti = X_test.iloc[:subset]
    yti = y_test.iloc[:subset]
    # Necesitamos datos ya transformados: usamos el pipeline y le pedimos al paso scaler
    Xti_scaled = pipe.named_steps["scaler"].transform(Xti)
    svc_model = pipe.named_steps["svc"]
    perm = permutation_importance(
        estimator=svc_model,
        X=Xti_scaled,
        y=yti,
        n_repeats=PERM_REPEATS,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    perm_imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    print("\n[TOP 15 FEATURES IMPORTANTES (Permutation)]")
    for i, (f, v) in enumerate(perm_imp.head(15).items(), 1):
        tag = "[NUEVA]" if f in features_mejoradas else "[ORIGINAL]"
        print(f"{i:2d}. {tag} {f}: {v:.6f}")

    # Guardados
    joblib.dump(pipe, "modelo_svc_final.pkl")
    perm_imp.to_csv("analisis_importancia_permutation_svc.csv", header=["importance"])
    print(f"\n[GUARDADO] Modelo final guardado en: modelo_svc_final.pkl")
    print(f"[GUARDADO] Importancias (Permutation) guardadas en: analisis_importancia_permutation_svc.csv")

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)

    # Resumen
    print(f"\n[RESUMEN DE TODAS LAS MEJORAS IMPLEMENTADAS]")
    print(f"[OK] Escalado con StandardScaler")
    print(f"[OK] SVC kernel RBF con class_weight='balanced'")
    print(f"[OK] Métricas Top-3/Top-5")
    print(f"[OK] Validación cruzada (train only)")
    print(f"[OK] Permutation Importance (subconjunto)")
    print(f"\n[ESTADISTICAS FINALES]")
    print(f"   Total de características: {len(features)}")
    print(f"   Características nuevas: {len(features_mejoradas)}")
    print(f"   Registros de entrenamiento: {len(X_train):,}")
    print(f"   Registros de prueba: {len(X_test):,}")
    print(f"   Tiempo de entrenamiento: {train_time:.2f} segundos")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
