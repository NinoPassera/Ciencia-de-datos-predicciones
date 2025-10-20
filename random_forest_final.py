#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest FINAL con TODAS las características adicionales
Incluye todas las mejoras para maximizar la precisión
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
from datetime import datetime

def main():
    print("=" * 70)
    print("RANDOM FOREST FINAL - TODAS LAS CARACTERISTICAS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    X = df[features_completas].fillna(0)
    y = df['destino'].astype(str)
    
    print(f"Features originales: {len(features_originales)}")
    print(f"Features mejoradas: {len(features_mejoradas)}")
    print(f"Total features: {len(features_completas)}")
    print(f"Destinos unicos: {y.nunique()}")
    
    # Dividir datos
    print("\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Entrenamiento: {len(X_train):,}")
    print(f"Prueba: {len(X_test):,}")
    
    # Entrenar modelo FINAL optimizado
    print("\nEntrenando Random Forest final optimizado...")
    modelo = RandomForestClassifier(
        n_estimators=300,           # Más árboles para mejor precisión
        max_depth=30,              # Más profundidad
        min_samples_split=2,        # Menos restricción
        min_samples_leaf=1,         # Menos restricción
        max_features='sqrt',       # Optimizar features por árbol
        bootstrap=True,            # Bootstrap para robustez
        random_state=42,
        n_jobs=-1
    )
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    print(f"[OK] Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")
    
    # Evaluar modelo
    print("\nEvaluando modelo final...")
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Validación cruzada para mejor evaluación
    print("\nRealizando validación cruzada...")
    cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
    
    print(f"\n[RESULTADOS FINALES]")
    print(f"Accuracy en test: {accuracy*100:.2f}%")
    print(f"Accuracy promedio CV: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Top 15 features importantes
    importances = pd.Series(modelo.feature_importances_, index=features_completas).sort_values(ascending=False)
    print(f"\n[TOP 15 FEATURES IMPORTANTES]")
    for i, (feature, importance) in enumerate(importances.head(15).items(), 1):
        es_nueva = "[NUEVA]" if feature in features_mejoradas else "[ORIGINAL]"
        print(f"{i:2d}. {es_nueva} {feature}: {importance:.4f}")
    
    # Análisis de nuevas características
    print(f"\n[ANALISIS DE NUEVAS CARACTERISTICAS]")
    nuevas_importances = importances[importances.index.isin(features_mejoradas)].sort_values(ascending=False)
    for feature, importance in nuevas_importances.items():
        ranking = list(importances.index).index(feature) + 1
        print(f"   {feature}: {importance:.4f} (ranking #{ranking})")
    
    # Comparación con modelos anteriores
    print(f"\n[COMPARACION CON MODELOS ANTERIORES]")
    print(f"   Modelo original: 47.01% accuracy")
    print(f"   Modelo mejorado: 49.97% accuracy")
    print(f"   Modelo final: {accuracy*100:.2f}% accuracy")
    
    mejora_original = accuracy - 0.4701
    mejora_mejorado = accuracy - 0.4997
    
    print(f"\n   Mejora vs original: +{mejora_original*100:.2f} puntos porcentuales")
    print(f"   Mejora vs mejorado: +{mejora_mejorado*100:.2f} puntos porcentuales")
    
    # Análisis de características más impactantes
    print(f"\n[ANALISIS DE IMPACTO]")
    top_5_nuevas = nuevas_importances.head(5)
    print(f"   Top 5 características nuevas más importantes:")
    for i, (feature, importance) in enumerate(top_5_nuevas.items(), 1):
        print(f"     {i}. {feature}: {importance:.4f}")
    
    # Guardar modelo final
    modelo_file = "modelo_random_forest_final.pkl"
    joblib.dump(modelo, modelo_file)
    print(f"\n[GUARDADO] Modelo final guardado en: {modelo_file}")
    
    # Guardar análisis de importancia
    importancia_file = "analisis_importancia_final.csv"
    importances_df = pd.DataFrame({
        'feature': importances.index,
        'importance': importances.values,
        'es_nueva': importances.index.isin(features_mejoradas),
        'ranking': range(1, len(importances) + 1)
    })
    importances_df.to_csv(importancia_file, index=False)
    print(f"[GUARDADO] Análisis de importancia final guardado en: {importancia_file}")
    
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
    print(f"[OK] Hiperparámetros optimizados")
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
