#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest FINAL (tunado) con TODAS las características
- Hiperparámetros: n_estimators=600, max_depth=32, min_samples_split=10, max_features=0.5
- Incluye: OOB score, Top-3/Top-5, CV=5, clasificación por clase, guardado de artefactos
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score

def main():
    print("=" * 70)
    print("RANDOM FOREST FINAL (TUNADO) - TODAS LAS CARACTERISTICAS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------- Controles de rendimiento --------------------
    MAX_ROWS    = 80000   # p.ej., 80000 para acelerar; None = usar todo
    RANDOM_SEED = 42
    CV_FOLDS    = 5
    # -----------------------------------------------------------------

    # Cargar datos
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

    X = df[features].fillna(0).astype(np.float32)  # float32 reduce a la mitad la RAM
    y = df['destino'].astype(str)


    print(f"\nFeatures originales: {len(features_originales)}")
    print(f"Features mejoradas: {len(features_mejoradas)}")
    print(f"Total features: {len(features)}")
    print(f"Destinos únicos: {y.nunique()}")

    # (Opcional) limitar filas con muestra estratificada para acelerar
    if MAX_ROWS is not None and MAX_ROWS < len(X):
        from sklearn.model_selection import StratifiedShuffleSplit
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

    # Modelo RF tunado (de tu búsqueda)
    modelo = RandomForestClassifier(
        n_estimators=600,
        max_depth=32,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        max_samples=0.8, 
        oob_score=True,          # métrica adicional sin CV
        class_weight=None,       # cambiar a 'balanced_subsample' si hay desbalance fuerte
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Entrenamiento
    print("\nEntrenando Random Forest final (tunado)...")
    t0 = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - t0
    print(f"[OK] Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")
    print(f"OOB score: {modelo.oob_score_*100:.2f}%")

    # Evaluación en test
    print("\nEvaluando modelo final...")
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\n[RESULTADOS EN TEST]")
    print(f"Accuracy: {acc*100:.2f}% | F1 macro: {f1m*100:.2f}%")

    # Top-k (muy útil con 89 clases)
    if hasattr(modelo, "predict_proba"):
        proba_test = modelo.predict_proba(X_test)
        top3 = top_k_accuracy_score(y_test, proba_test, k=3, labels=modelo.classes_)
        top5 = top_k_accuracy_score(y_test, proba_test, k=5, labels=modelo.classes_)
        print(f"Top-3 accuracy: {top3*100:.2f}% | Top-5 accuracy: {top5*100:.2f}%")

    print("\n[Classification report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Validación cruzada (accuracy) sobre TODO X,y para reportar robustez
    print("\nRealizando validación cruzada (CV=5)...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy promedio CV: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

    # Importancias
    importances = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n[TOP 15 FEATURES IMPORTANTES]")
    for i, (feature, importance) in enumerate(importances.head(15).items(), 1):
        tag = "[NUEVA]" if feature in features_mejoradas else "[ORIGINAL]"
        print(f"{i:2d}. {tag} {feature}: {importance:.4f}")

    # Guardados
    model_file = "modelo_random_forest_final_tunado.pkl"
    importances_file = "analisis_importancia_final_tunado.csv"
    joblib.dump(modelo, model_file)
    importances.to_csv(importances_file, header=["importance"])
    print(f"\n[GUARDADO] Modelo final guardado en: {model_file}")
    print(f"[GUARDADO] Importancias guardadas en: {importances_file}")

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)

    # Resumen
    print(f"\n[RESUMEN]")
    print(f"[OK] Hiperparámetros tunados aplicados")
    print(f"[OK] OOB score reportado")
    print(f"[OK] Métricas: Accuracy, F1 macro, Top-3/Top-5")
    print(f"[OK] Validación cruzada (CV={CV_FOLDS})")
    print(f"\n[ESTADISTICAS FINALES]")
    print(f"   Total de características: {len(features)}")
    print(f"   Características nuevas: {len(features_mejoradas)}")
    print(f"   Registros de entrenamiento: {len(X_train):,}")
    print(f"   Registros de prueba: {len(X_test):,}")
    print(f"   Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
