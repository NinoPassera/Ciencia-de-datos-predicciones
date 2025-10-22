#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression FINAL con TODAS las características adicionales
Incluye escalado, validación cruzada y análisis de coeficientes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import time
from datetime import datetime

def main():
    print("=" * 70)
    print("LOGISTIC REGRESSION FINAL - TODAS LAS CARACTERISTICAS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------- CONTROLES OPCIONALES DE RENDIMIENTO --------------------
    MAX_ROWS    = 120000   #limitar filas estratificadas. None = usar todo
    RANDOM_SEED = 42
    CV_FOLDS    = 5
    # ---------------------------------------------------------------------------

    # Cargar datos finales
    print("\nCargando dataset final...")
    try:
        df = pd.read_csv("dataset_modelo_final.csv")
        print(f"[OK] Dataset final cargado: {len(df):,} registros")
    except FileNotFoundError:
        print("[ERROR] No se encontró dataset_modelo_final.csv")
        print("Ejecuta primero: python crear_dataset_final.py")
        return

    # Features completas
    features_originales = [
        'origen_lat', 'origen_lon',
        'hora_salida', 'dia_semana', 'mes',
        'viajes_totales', 'semanas_activas', 'viajes_por_semana', 'duracion_promedio_min'
    ]

    features_mejoradas = [
        'periodo_dia_numerico',      # 1: mañana, 2: tarde, 3: noche, 0: madrugada
        'es_fin_semana',            # 1: fin de semana, 0: día laboral
        'es_hora_pico',             # 1: hora pico, 0: hora normal
        'zona_origen',              # 1: centro, 2: cerca, 3: periferia, 4: lejos
        'capacidad_origen',         # Capacidad de la estación origen
        'estaciones_cercanas_origen', # Estaciones cercanas al origen
        'variedad_destinos',        # Número de destinos únicos del usuario
        'variedad_origenes',        # Número de orígenes únicos del usuario
        'consistencia_horaria',     # Desviación estándar de horas de viaje
        'distancia_promedio_usuario', # Distancia promedio de viajes del usuario
        'dia_favorito',             # Día favorito de la semana del usuario
        'frecuencia_lunes', 'frecuencia_martes', 'frecuencia_miercoles',
        'frecuencia_jueves', 'frecuencia_viernes', 'frecuencia_sabado', 'frecuencia_domingo',
    ]

    features_completas = features_originales + features_mejoradas

    X = df[features_completas].fillna(0).astype(np.float32)  # float32 para ahorrar RAM
    y = df['destino'].astype(str)

    print(f"Features originales: {len(features_originales)}")
    print(f"Features mejoradas: {len(features_mejoradas)}")
    print(f"Total features: {len(features_completas)}")
    print(f"Destinos únicos: {y.nunique()}")

    # (OPCIONAL) Limitar filas con muestra estratificada por destino
    if MAX_ROWS is not None and MAX_ROWS < len(X):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_ROWS, random_state=RANDOM_SEED)
        idx_sample, _ = next(sss.split(X, y))
        X = X.iloc[idx_sample].reset_index(drop=True)
        y = y.iloc[idx_sample].reset_index(drop=True)
        print(f"[MUESTRA] Usando muestra estratificada de {len(X):,} filas")

    # Dividir datos
    print("\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"Entrenamiento: {len(X_train):,}")
    print(f"Prueba: {len(X_test):,}")

    # Escalado (CRÍTICO para Logistic Regression)
    print("\nNormalizando datos...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Entrenar modelo FINAL
    # Elecciones razonables: lbfgs (multinomial), l2, C intermedio, max_iter alto y balanceo opcional
    print("\nEntrenando Logistic Regression final optimizado...")
    modelo = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        C=1.0,
        max_iter=8000,
        class_weight='balanced',   # útil si hay desbalance entre destinos
        multi_class='auto',
        n_jobs=None,
        random_state=RANDOM_SEED
    )

    inicio = time.time()
    modelo.fit(X_train_scaled, y_train)
    tiempo_entrenamiento = time.time() - inicio
    print(f"[OK] Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")

    # Evaluar modelo
    print("\nEvaluando modelo final...")
    y_pred = modelo.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n[RESULTADOS FINALES]")
    print(f"Accuracy en test: {accuracy*100:.2f}%")
    print("\n[Classification report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Validación cruzada (con pipeline interno para evitar fuga de info)
    print("\nRealizando validación cruzada...")
    # Reentrenamos en CV con escalado dentro de cada fold para no filtrar info
    from sklearn.pipeline import Pipeline
    pipe_cv = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            C=1.0,
            max_iter=8000,
            class_weight='balanced',
            multi_class='auto',
            random_state=RANDOM_SEED
        ))
    ])
    cv_scores = cross_val_score(pipe_cv, X, y, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy promedio CV: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

    # Importancia de features (promedio de |coef| por feature en multiclase)
    print(f"\n[TOP 15 FEATURES IMPORTANTES (|coef| promedio)]")
    if hasattr(modelo, "coef_"):
        coef_abs_mean = np.mean(np.abs(modelo.coef_), axis=0)
        importances = pd.Series(coef_abs_mean, index=features_completas).sort_values(ascending=False)
        for i, (feature, importance) in enumerate(importances.head(15).items(), 1):
            es_nueva = "[NUEVA]" if feature in features_mejoradas else "[ORIGINAL]"
            print(f"{i:2d}. {es_nueva} {feature}: {importance:.4f}")

        # Análisis de nuevas características
        print(f"\n[ANALISIS DE NUEVAS CARACTERISTICAS]")
        nuevas_importances = importances[importances.index.isin(features_mejoradas)].sort_values(ascending=False)
        for feature, importance in nuevas_importances.items():
            ranking = list(importances.index).index(feature) + 1
            print(f"   {feature}: {importance:.4f} (ranking #{ranking})")

        # Guardar análisis de coeficientes
        importancia_file = "analisis_coeficientes_logistic_final.csv"
        coef_df = pd.DataFrame({
            'feature': importances.index,
            'importance_abs_mean_coef': importances.values,
            'es_nueva': importances.index.isin(features_mejoradas),
            'ranking': range(1, len(importances) + 1)
        })
        coef_df.to_csv(importancia_file, index=False)
        print(f"[GUARDADO] Análisis de coeficientes guardado en: {importancia_file}")

    # Guardar modelo y scaler
    modelo_file = "modelo_logistic_regression_final.pkl"
    scaler_file = "scaler_logistic_regression_final.pkl"
    joblib.dump(modelo, modelo_file)
    joblib.dump(scaler, scaler_file)
    print(f"\n[GUARDADO] Modelo final guardado en: {modelo_file}")
    print(f"[GUARDADO] Scaler guardado en: {scaler_file}")

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)

    # Resumen de todas las mejoras
    print(f"\n[RESUMEN DE TODAS LAS MEJORAS IMPLEMENTADAS]")
    print(f"[OK] Período del día numérico (0-3)")
    print(f"[OK] Indicador de fin de semana")
    print(f"[OK] Indicador de hora pico")
    print(f"[OK] Zona geográfica de origen")
    print(f"[OK] Capacidad de estación origen")
    print(f"[OK] Estaciones cercanas al origen")
    print(f"[OK] Variedad de destinos del usuario")
    print(f"[OK] Variedad de orígenes del usuario")
    print(f"[OK] Consistencia horaria del usuario")
    print(f"[OK] Distancia promedio del usuario")
    print(f"[OK] Día favorito de la semana")
    print(f"[OK] Frecuencias por día de la semana (7 características)")
    print(f"[OK] Normalización de datos")
    print(f"[OK] Validación cruzada implementada")

    print(f"\n[ESTADISTICAS FINALES]")
    print(f"   Total de características: {len(features_completas)}")
    print(f"   Características nuevas: {len(features_mejoradas)}")
    print(f"   Registros de entrenamiento: {len(X_train):,}")
    print(f"   Registros de prueba: {len(X_test):,}")
    print(f"   Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
