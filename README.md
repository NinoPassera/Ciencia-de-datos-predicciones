# Sistema de Predicción de Destinos en Bicicleta

Este proyecto implementa un sistema de machine learning para predecir destinos de viajes en bicicleta basado en datos históricos de usuarios y características temporales.

## Descripción

El sistema analiza patrones de viajes en bicicleta y utiliza un modelo Random Forest para predecir el destino más probable de un viaje basado en:
- Coordenadas de origen
- Hora del día
- Día de la semana
- Mes del año
- Historial del usuario (viajes totales, frecuencia, duración promedio)

## Archivos del Proyecto

### Scripts Principales

- **`crear_dataset_sin_emojis.py`**: Script para procesar datos de viajes y crear el dataset de entrenamiento
- **`sistema_prediccion_ejecutable.py`**: Sistema completo de entrenamiento y predicción

### Datos Requeridos

Para ejecutar el sistema, necesitas los siguientes archivos de datos:
- `trips_2024-09-09_to_2025-09-09 (1).csv`: Datos históricos de viajes
- `station_data_enriched (1).csv`: Información de estaciones de bicicletas

## Uso

### 1. Crear el Dataset

```bash
python crear_dataset_sin_emojis.py
```

Este script:
- Carga y procesa los datos de viajes
- Limpia y normaliza el texto
- Calcula métricas de usuario
- Integra coordenadas de estaciones
- Genera `dataset_modelo_stream.csv`

### 2. Entrenar Modelo y Hacer Predicciones

```bash
python sistema_prediccion_ejecutable.py
```

Este script:
- Entrena un modelo Random Forest
- Evalúa el rendimiento del modelo
- Ejecuta ejemplos de predicción
- Guarda el modelo entrenado

## Resultados del Modelo

- **Accuracy**: 47.01%
- **Top-3 accuracy**: 65.62%
- **Top-5 accuracy**: 72.03%

### Variables Más Importantes

1. Duración promedio del usuario (16.25%)
2. Longitud de origen (15.55%)
3. Latitud de origen (14.68%)
4. Hora de salida (10.68%)
5. Viajes totales (10.32%)

## Características Técnicas

- **Algoritmo**: Random Forest Classifier
- **Features**: 9 variables numéricas
- **Destinos**: 89 estaciones únicas
- **Datos de entrenamiento**: ~150,000 registros

## Ejemplos de Predicción

El sistema incluye 3 ejemplos predefinidos que demuestran diferentes escenarios:
1. Viaje desde Plaza San Martín (lunes 8:00 AM)
2. Viaje desde Alameda (viernes 6:00 PM)
3. Viaje desde Estación Central (miércoles 12:00 PM)

## Requisitos

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

## Instalación

```bash
pip install pandas numpy scikit-learn joblib
```

## Autor

Proyecto de ciencia de datos para predicción de destinos en sistemas de bicicletas compartidas.
