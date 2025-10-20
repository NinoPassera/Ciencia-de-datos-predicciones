# Sistema de Predicción de Destinos en Bicicleta 🚴‍♂️

Este proyecto implementa un sistema avanzado de machine learning para predecir destinos de viajes en bicicleta basado en datos históricos de usuarios, características temporales, geográficas y patrones de comportamiento.

## 🎯 Descripción

El sistema analiza patrones complejos de viajes en bicicleta y utiliza un modelo Random Forest optimizado para predecir el destino más probable de un viaje basado en **27 características** que incluyen:

### Características Originales (9)
- Coordenadas de origen (latitud/longitud)
- Hora del día, día de la semana, mes del año
- Historial del usuario (viajes totales, frecuencia, duración promedio)

### Características Mejoradas (18)
- **Temporales**: Período del día, fin de semana, hora pico
- **Geográficas**: Zona de origen, capacidad de estación, estaciones cercanas
- **Comportamiento**: Variedad de destinos/orígenes, consistencia horaria, distancia promedio
- **Patrones semanales**: Día favorito, frecuencias por día de la semana

## 📊 Resultados del Modelo

- **Accuracy**: **53.66%** (mejora de +6.65% vs modelo original)
- **Validación cruzada**: 47.02% (+/- 2.58%)
- **Tiempo de entrenamiento**: ~27 segundos
- **Destinos únicos**: 89 estaciones
- **Registros de entrenamiento**: 150,064

### 🔥 Características Más Importantes

1. **Distancia promedio del usuario** (10.40%) - ¡La más predictiva!
2. **Mes del año** (6.62%)
3. **Hora de salida** (6.50%)
4. **Longitud de origen** (5.82%)
5. **Duración promedio** (5.82%)

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

### Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn joblib
```

### Datos Requeridos

Asegúrate de tener estos archivos en el directorio del proyecto:
- `trips_2024-09-09_to_2025-09-09 (1).csv` - Datos históricos de viajes
- `station_data_enriched (1).csv` - Información de estaciones de bicicletas

### Ejecución del Proyecto

#### Paso 1: Crear el Dataset Final
```bash
python crear_dataset_final.py
```

Este script:
- ✅ Carga y procesa 159,155 registros de viajes
- ✅ Limpia y normaliza texto de estaciones
- ✅ Calcula métricas avanzadas de usuario
- ✅ Integra coordenadas y zonas geográficas
- ✅ Genera `dataset_modelo_final.csv` con 27 características

#### Paso 2: Entrenar el Modelo
```bash
python random_forest_final.py
```

Este script:
- ✅ Entrena modelo Random Forest optimizado
- ✅ Evalúa rendimiento con validación cruzada
- ✅ Genera análisis de importancia de características
- ✅ Guarda modelo entrenado y análisis

## 📁 Archivos del Proyecto

### Scripts Principales
- **`crear_dataset_final.py`** - Procesamiento completo de datos
- **`random_forest_final.py`** - Entrenamiento del modelo final

### Datos de Entrada
- **`trips_2024-09-09_to_2025-09-09 (1).csv`** - Datos de viajes
- **`station_data_enriched (1).csv`** - Datos de estaciones

### Resultados Generados
- **`dataset_modelo_final.csv`** - Dataset procesado (150,064 registros × 28 columnas)
- **`modelo_random_forest_final.pkl`** - Modelo entrenado (~16GB)
- **`analisis_importancia_final.csv`** - Análisis de características

## 🔧 Características Técnicas

- **Algoritmo**: Random Forest Classifier optimizado
- **Features**: 27 variables numéricas
- **Hiperparámetros**: n_estimators=300, max_depth=30, max_features='sqrt'
- **Validación**: 5-fold cross-validation
- **Destinos**: 89 estaciones únicas
- **Datos**: 150,064 registros de entrenamiento

## 📈 Evolución del Modelo

| Versión | Accuracy | Mejora | Características |
|---------|----------|--------|-----------------|
| Original | 47.01% | - | 9 características |
| Mejorado | 49.97% | +2.96% | 17 características |
| **Final** | **53.66%** | **+6.65%** | **27 características** |

## 🎯 Ejemplo de Uso del Modelo

```python
import joblib
import pandas as pd

# Cargar modelo entrenado
modelo = joblib.load('modelo_random_forest_final.pkl')

# Preparar datos de entrada
datos_usuario = {
    'origen_lat': -32.88718,
    'origen_lon': -68.84085,
    'hora_salida': 8,  # 8:00 AM
    'dia_semana': 0,   # Lunes
    'mes': 3,          # Marzo
    'viajes_totales': 45,
    'distancia_promedio_usuario': 0.025,
    # ... resto de características
}

# Hacer predicción
X_nuevo = pd.DataFrame([datos_usuario])
destino_predicho = modelo.predict(X_nuevo)[0]
probabilidades = modelo.predict_proba(X_nuevo)[0]

print(f"Destino predicho: {destino_predicho}")
print(f"Probabilidad: {max(probabilidades)*100:.2f}%")
```

## 🔍 Análisis de Patrones Encontrados

### Patrones Temporales
- **Hora pico**: 7-9 AM y 5-7 PM tienen patrones específicos
- **Fin de semana**: Comportamiento diferente vs días laborales
- **Lunes**: Día más activo (27,773 viajes)

### Patrones Geográficos
- **Zona centro**: Más conexiones con estaciones cercanas
- **Capacidad de estación**: Estaciones grandes atraen más tráfico
- **Distancia promedio**: Cada usuario tiene un "radio de acción" típico

### Patrones de Usuario
- **Consistencia horaria**: Usuarios muy predecibles en horarios
- **Variedad**: Usuarios con más variedad son más predecibles
- **Comportamiento semanal**: Patrones específicos por día

## 🛠️ Solución de Problemas

### Error: "No se encontró el archivo"
- Verifica que los archivos CSV de datos estén en el directorio correcto
- Asegúrate de que los nombres de archivo coincidan exactamente

### Error de memoria
- El modelo final es grande (~16GB), asegúrate de tener suficiente RAM
- Considera usar un entorno con al menos 8GB de RAM disponible

### Error de dependencias
```bash
pip install --upgrade pandas numpy scikit-learn joblib
```

## 📊 Estructura del Dataset Final

El dataset final contiene 27 características organizadas en:

1. **Geográficas**: origen_lat, origen_lon, zona_origen, capacidad_origen, estaciones_cercanas_origen
2. **Temporales**: hora_salida, dia_semana, mes, periodo_dia_numerico, es_fin_semana, es_hora_pico
3. **Usuario**: viajes_totales, semanas_activas, viajes_por_semana, duracion_promedio_min, distancia_promedio_usuario
4. **Comportamiento**: variedad_destinos, variedad_origenes, consistencia_horaria, dia_favorito
5. **Frecuencias semanales**: frecuencia_lunes, frecuencia_martes, ..., frecuencia_domingo

## 🎓 Autor

Proyecto de ciencia de datos para predicción de destinos en sistemas de bicicletas compartidas.

**Mejoras implementadas**: +18 características nuevas, +6.65% de accuracy, análisis completo de importancia de características.