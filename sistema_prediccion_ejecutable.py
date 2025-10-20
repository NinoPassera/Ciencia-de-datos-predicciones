#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de predicción de destinos ejecutable
Entrena el modelo y hace predicciones directamente
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score
import joblib
import os

def entrenar_modelo():
    """Entrenar el modelo con el dataset disponible"""
    print("Entrenando modelo de predicción de destinos...")
    
    # Verificar que existe el dataset
    dataset_file = "dataset_modelo_stream.csv"
    if not os.path.exists(dataset_file):
        print(f"Error: No se encontró el archivo {dataset_file}")
        print("Ejecuta primero crear_dataset_completo.py")
        return None
    
    # Cargar dataset
    print("Cargando dataset...")
    df = pd.read_csv(dataset_file)
    print(f"Dataset cargado: {len(df):,} registros")
    
    # Preparar features y target
    features = [
        'origen_lat', 'origen_lon',
        'hora_salida', 'dia_semana', 'mes',
        'viajes_totales', 'semanas_activas', 'viajes_por_semana', 'duracion_promedio_min'
    ]
    
    X = df[features].fillna(0)
    y = df['destino'].astype(str)
    
    print(f"Features: {features}")
    print(f"Destinos únicos: {y.nunique()}")
    print(f"Registros para entrenamiento: {len(X):,}")
    
    # Dividir datos
    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Datos de entrenamiento: {len(X_train):,}")
    print(f"Datos de prueba: {len(X_test):,}")
    
    # Entrenar modelo
    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    print("Evaluando modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Top-k accuracy
    proba = model.predict_proba(X_test)
    top3_accuracy = top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
    top5_accuracy = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)
    
    print(f"Top-3 accuracy: {top3_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")
    
    # Guardar modelo
    model_file = "modelo_prediccion_destinos.pkl"
    joblib.dump(model, model_file)
    print(f"Modelo guardado en: {model_file}")
    
    # Importancia de variables
    print("Importancia de variables:")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feature, importance in importances.items():
        print(f"  {feature}: {importance:.4f}")
    
    return model

def cargar_modelo():
    """Cargar el modelo entrenado"""
    model_file = "modelo_prediccion_destinos.pkl"
    if not os.path.exists(model_file):
        print(f"Modelo no encontrado en {model_file}")
        return None
    
    print(f"Cargando modelo desde {model_file}...")
    model = joblib.load(model_file)
    print("Modelo cargado exitosamente")
    return model

def hacer_prediccion(model, origen_lat, origen_lon, hora_salida, dia_semana, mes, 
                    viajes_totales=0, semanas_activas=0, viajes_por_semana=0, duracion_promedio=0):
    """
    Hacer predicción de destino para un viaje
    """
    # Crear DataFrame con los datos de entrada
    datos = pd.DataFrame({
        'origen_lat': [origen_lat],
        'origen_lon': [origen_lon],
        'hora_salida': [hora_salida],
        'dia_semana': [dia_semana],
        'mes': [mes],
        'viajes_totales': [viajes_totales],
        'semanas_activas': [semanas_activas],
        'viajes_por_semana': [viajes_por_semana],
        'duracion_promedio_min': [duracion_promedio]
    })
    
    # Hacer predicción
    prediccion = model.predict(datos)[0]
    probabilidades = model.predict_proba(datos)[0]
    
    # Obtener top 5 predicciones
    clases = model.classes_
    indices_top5 = np.argsort(probabilidades)[-5:][::-1]
    
    print(f"Prediccion principal: {prediccion}")
    print(f"Top 5 predicciones:")
    for i, idx in enumerate(indices_top5):
        print(f"  {i+1}. {clases[idx]} (probabilidad: {probabilidades[idx]:.3f})")
    
    return prediccion, probabilidades, clases

def ejecutar_ejemplos():
    """Ejecutar ejemplos de predicción"""
    print("=== EJEMPLOS DE PREDICCION DE DESTINO ===\n")
    
    # Siempre entrenar un nuevo modelo para mostrar resultados
    print("Entrenando nuevo modelo con el dataset actualizado...")
    model = entrenar_modelo()
    if model is None:
        return
    
    # Ejemplo 1: Viaje desde Plaza San Martín a las 8:00 AM un lunes en enero
    print("\nEjemplo 1: Viaje desde Plaza San Martin (lunes 8:00 AM, enero)")
    print("Coordenadas: -32.88718, -68.84085")
    print("Usuario con 50 viajes totales, 8 semanas activas")
    
    hacer_prediccion(
        model=model,
        origen_lat=-32.88718,
        origen_lon=-68.84085,
        hora_salida=8,
        dia_semana=0,  # lunes
        mes=1,         # enero
        viajes_totales=50,
        semanas_activas=8,
        viajes_por_semana=6.25,
        duracion_promedio=15
    )
    
    print("\n" + "="*60)
    
    # Ejemplo 2: Viaje desde Alameda a las 18:00 un viernes en marzo
    print("\nEjemplo 2: Viaje desde Alameda (viernes 6:00 PM, marzo)")
    print("Coordenadas: -32.88167, -68.83681")
    print("Usuario nuevo (pocos viajes)")
    
    hacer_prediccion(
        model=model,
        origen_lat=-32.88167,
        origen_lon=-68.83681,
        hora_salida=18,
        dia_semana=4,  # viernes
        mes=3,         # marzo
        viajes_totales=5,
        semanas_activas=2,
        viajes_por_semana=2.5,
        duracion_promedio=20
    )
    
    print("\n" + "="*60)
    
    # Ejemplo 3: Viaje desde estación central a las 12:00 un miércoles en junio
    print("\nEjemplo 3: Viaje desde Estacion Central (miercoles 12:00 PM, junio)")
    print("Coordenadas: -32.88472, -68.84556")
    print("Usuario activo (100+ viajes)")
    
    hacer_prediccion(
        model=model,
        origen_lat=-32.88472,
        origen_lon=-68.84556,
        hora_salida=12,
        dia_semana=2,  # miércoles
        mes=6,         # junio
        viajes_totales=120,
        semanas_activas=15,
        viajes_por_semana=8.0,
        duracion_promedio=12
    )

def main():
    """Función principal"""
    print("Sistema de Prediccion de Destinos en Bicicleta")
    print("=" * 50)
    
    # Verificar si existe el dataset
    if not os.path.exists("dataset_modelo_stream.csv"):
        print("Error: No se encontro dataset_modelo_stream.csv")
        print("Ejecuta primero crear_dataset_completo.py para crear el dataset")
        return
    
    # Ejecutar ejemplos
    ejecutar_ejemplos()
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error durante la ejecucion: {e}")
        import traceback
        traceback.print_exc()
