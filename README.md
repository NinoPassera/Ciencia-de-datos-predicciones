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

## 📋 Explicación Detallada de Variables

### 🗺️ Variables Geográficas (5 variables)

#### **1. `origen_lat` y `origen_lon`**
- **Qué es**: Coordenadas geográficas de la estación de origen
- **Cómo se calcula**: Se extrae directamente de `station_data_enriched.csv`
- **Ejemplo**: `-32.89142, -68.86011` (Plaza San Martín)
- **Importancia**: Muy alta (ranking #6 y #4) - La ubicación es fundamental

#### **3. `zona_origen`**
- **Qué es**: Clasificación geográfica de la estación origen
- **Cómo se calcula**:
```python
def clasificar_zona_geografica(lat, lon):
    centro_lat, centro_lon = -32.89, -68.84  # Centro de Mendoza
    dist_lat = abs(lat - centro_lat)
    dist_lon = abs(lon - centro_lon)
    
    if dist_lat < 0.02 and dist_lon < 0.02: return 1    # Centro
    elif dist_lat < 0.05 and dist_lon < 0.05: return 2 # Cerca del centro
    elif dist_lat < 0.1 and dist_lon < 0.1: return 3   # Periferia
    else: return 4  # Lejos del centro
```
- **Valores**: 1=Centro, 2=Cerca, 3=Periferia, 4=Lejos

#### **4. `capacidad_origen`**
- **Qué es**: Número de bicicletas que puede albergar la estación origen
- **Cómo se calcula**: Se extrae de `station_capacity` en `station_data_enriched.csv`
- **Ejemplo**: 15 bicicletas máximo
- **Importancia**: Alta (ranking #8) - Estaciones grandes atraen más tráfico

#### **5. `estaciones_cercanas_origen`**
- **Qué es**: Número de estaciones en un radio de 1km del origen
- **Cómo se calcula**:
```python
# Para cada estación origen, contar estaciones en radio de 0.01 grados (~1km)
distances = calcular_distancia_euclidiana(origen_lat, origen_lon, otras_estaciones)
estaciones_cercanas = (distances <= 0.01).sum() - 1  # -1 para excluir la misma estación
```

### ⏰ Variables Temporales (6 variables)

#### **6. `hora_salida`**
- **Qué es**: Hora del día cuando inicia el viaje (0-23)
- **Cómo se calcula**: `df["Fecha origen"].dt.hour`
- **Ejemplo**: 8 = 8:00 AM, 14 = 2:00 PM
- **Importancia**: Muy alta (ranking #3)

#### **7. `dia_semana`**
- **Qué es**: Día de la semana (0=Lunes, 6=Domingo)
- **Cómo se calcula**: `df["Fecha origen"].dt.dayofweek`
- **Ejemplo**: 0=Lunes, 5=Sábado, 6=Domingo
- **Importancia**: Alta (ranking #7)

#### **8. `mes`**
- **Qué es**: Mes del año (1-12)
- **Cómo se calcula**: `df["Fecha origen"].dt.month`
- **Ejemplo**: 1=Enero, 12=Diciembre
- **Importancia**: Muy alta (ranking #2) - Patrones estacionales

#### **9. `periodo_dia_numerico`**
- **Qué es**: Período del día clasificado numéricamente
- **Cómo se calcula**:
```python
def clasificar_periodo_numerico(hora):
    if 6 <= hora < 12: return 1    # mañana
    elif 12 <= hora < 18: return 2 # tarde  
    elif 18 <= hora < 24: return 3 # noche
    else: return 0                 # madrugada
```
- **Valores**: 0=Madrugada, 1=Mañana, 2=Tarde, 3=Noche

#### **10. `es_fin_semana`**
- **Qué es**: Indicador binario de fin de semana
- **Cómo se calcula**: `df["dia_semana"].isin([5, 6]).astype(int)`
- **Valores**: 0=Día laboral, 1=Fin de semana (Sábado/Domingo)

#### **11. `es_hora_pico`**
- **Qué es**: Indicador binario de hora pico
- **Cómo se calcula**: `df["hora_salida"].isin([7, 8, 9, 17, 18, 19]).astype(int)`
- **Valores**: 0=Hora normal, 1=Hora pico (7-9AM, 5-7PM)

### 👤 Variables de Usuario (6 variables)

#### **12. `viajes_totales`**
- **Qué es**: Número total de viajes que ha hecho el usuario
- **Cómo se calcula**: `df.groupby("Usuario_key").size()`
- **Ejemplo**: 45 viajes totales
- **Importancia**: Alta (ranking #10)

#### **13. `semanas_activas`**
- **Qué es**: Número de semanas diferentes en que el usuario ha usado el servicio
- **Cómo se calcula**: `df.groupby("Usuario_key")["semana"].nunique()`
- **Ejemplo**: 12 semanas activas

#### **14. `viajes_por_semana`**
- **Qué es**: Frecuencia promedio de viajes por semana
- **Cómo se calcula**: `viajes_totales / semanas_activas`
- **Ejemplo**: 3.75 viajes por semana

#### **15. `duracion_promedio_min`**
- **Qué es**: Duración promedio de viajes del usuario en minutos
- **Cómo se calcula**: `df.groupby("Usuario_key")["dur_min"].mean()`
- **Ejemplo**: 18.5 minutos promedio

#### **16. `distancia_promedio_usuario`** ⭐ **LA MÁS IMPORTANTE**
- **Qué es**: Distancia promedio que recorre el usuario en sus viajes
- **Cómo se calcula**:
```python
# 1. Calcular distancia de cada viaje
distancia_viaje = sqrt((destino_lat - origen_lat)² + (destino_lon - origen_lon)²)

# 2. Promedio por usuario
distancia_promedio_usuario = df.groupby("Usuario_key")["distancia_viaje"].mean()
```
- **Ejemplo**: 0.025 grados promedio
- **Importancia**: SÚPER ALTA (ranking #1) - Cada usuario tiene un "radio de acción" típico

#### **17. `consistencia_horaria`**
- **Qué es**: Qué tan consistente es el usuario con sus horarios de viaje
- **Cómo se calcula**: `df.groupby("Usuario_key")["hora_salida"].std()`
- **Ejemplo**: 2.1 (desviación estándar baja = muy consistente)
- **Importancia**: Alta (ranking #9)

### 🎯 Variables de Comportamiento (4 variables)

#### **18. `variedad_destinos`**
- **Qué es**: Número de destinos únicos que visita el usuario
- **Cómo se calcula**: `df.groupby("Usuario_key")["Destino_clean"].nunique()`
- **Ejemplo**: 12 destinos diferentes
- **Importancia**: Moderada (ranking #13)

#### **19. `variedad_origenes`**
- **Qué es**: Número de orígenes únicos que usa el usuario
- **Cómo se calcula**: `df.groupby("Usuario_key")["Origen_clean"].nunique()`
- **Ejemplo**: 5 orígenes diferentes
- **Importancia**: Moderada (ranking #14)

#### **20. `dia_favorito`**
- **Qué es**: Día de la semana favorito del usuario
- **Cómo se calcula**:
```python
# Calcular frecuencias por día
frecuencias_dias = df.groupby("Usuario_key")["dia_semana"].value_counts()

# El día con más viajes es el favorito
dia_favorito = frecuencias_dias.idxmax()
```
- **Valores**: 0=Lunes, 1=Martes, ..., 6=Domingo

### 📅 Variables de Frecuencia Semanal (7 variables)

#### **21-27. `frecuencia_lunes` a `frecuencia_domingo`**
- **Qué es**: Número de viajes que hace el usuario cada día de la semana
- **Cómo se calcula**:
```python
frecuencia_lunes = df.groupby("Usuario_key")["dia_semana"].apply(lambda x: (x == 0).sum())
frecuencia_martes = df.groupby("Usuario_key")["dia_semana"].apply(lambda x: (x == 1).sum())
# ... y así para cada día
```
- **Ejemplo**: Usuario hace 8 viajes los lunes, 6 los martes, etc.
- **Importancia**: Moderada (rankings #16-23)

### 🔍 Ejemplo Práctico de Cálculo

```python
# Usuario: María
usuario_key = "MARIA_GONZALEZ"

# Sus datos históricos:
viajes_maria = df[df["Usuario_key"] == usuario_key]

# Cálculos:
viajes_totales = len(viajes_maria)  # 45
semanas_activas = viajes_maria["semana"].nunique()  # 12
viajes_por_semana = viajes_totales / semanas_activas  # 3.75

duracion_promedio = viajes_maria["dur_min"].mean()  # 18.5
distancia_promedio = viajes_maria["distancia_viaje"].mean()  # 0.025
consistencia_horaria = viajes_maria["hora_salida"].std()  # 2.1

variedad_destinos = viajes_maria["Destino_clean"].nunique()  # 12
variedad_origenes = viajes_maria["Origen_clean"].nunique()  # 5

frecuencia_lunes = (viajes_maria["dia_semana"] == 0).sum()  # 8
frecuencia_martes = (viajes_maria["dia_semana"] == 1).sum()  # 6
# ... etc
```

### 🎯 ¿Por qué estas variables son tan predictivas?

1. **`distancia_promedio_usuario`**: Cada persona tiene un "radio de acción" típico
2. **`mes`**: Patrones estacionales (más viajes en primavera/verano)
3. **`hora_salida`**: Rutinas diarias (trabajo, casa, etc.)
4. **`consistencia_horaria`**: Usuarios rutinarios son más predecibles
5. **`capacidad_origen`**: Estaciones grandes atraen más tráfico

¡Estas 27 variables capturan patrones muy específicos del comportamiento humano en sistemas de transporte!

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