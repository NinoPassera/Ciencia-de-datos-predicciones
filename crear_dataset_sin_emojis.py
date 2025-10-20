#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script completo para crear el dataset_modelo_stream.csv
Hace todo el análisis, procesamiento y creación del dataset final
"""

import pandas as pd
import numpy as np
import unicodedata
import os
from sklearn.model_selection import train_test_split

def main():
    print("=" * 60)
    print("CREADOR DE DATASET MODELO STREAM - ANALISIS COMPLETO")
    print("=" * 60)
    
    # Rutas de archivos locales
    PATH_VIAJES = "trips_2024-09-09_to_2025-09-09 (1).csv"
    PATH_ESTACIONES = "station_data_enriched (1).csv"
    
    # Verificar que los archivos existen
    if not os.path.exists(PATH_VIAJES):
        print(f"Error: No se encontro el archivo {PATH_VIAJES}")
        return
    
    if not os.path.exists(PATH_ESTACIONES):
        print(f"Error: No se encontro el archivo {PATH_ESTACIONES}")
        return
    
    print("Archivos encontrados, iniciando procesamiento...")
    
    # ============================
    # PASO 1: Cargar datos
    # ============================
    print("\nCargando datos...")
    
    # Cargar datos de viajes (solo columnas necesarias para optimizar memoria)
    print("   - Cargando datos de viajes...")
    df_viajes = pd.read_csv(PATH_VIAJES, usecols=[
        "Usuario", "Origen", "Destino", "Fecha origen", "Fecha destino"
    ])
    print(f"     Registros cargados: {len(df_viajes):,}")
    
    # Cargar datos de estaciones
    print("   - Cargando datos de estaciones...")
    df_estaciones = pd.read_csv(PATH_ESTACIONES, usecols=[
        "station_name", "station_lat", "station_lon"
    ]).drop_duplicates("station_name")
    print(f"     Estaciones cargadas: {df_estaciones['station_name'].nunique()}")
    
    # ============================
    # PASO 2: Procesamiento de datos de viajes
    # ============================
    print("\nProcesando datos de viajes...")
    
    df = df_viajes.copy()
    
    # Parseo de fechas
    print("   - Parseando fechas...")
    df["Fecha origen"] = pd.to_datetime(df["Fecha origen"], errors="coerce", dayfirst=True)
    df["Fecha destino"] = pd.to_datetime(df["Fecha destino"], errors="coerce", dayfirst=True)
    
    # Duración en minutos
    print("   - Calculando duración de viajes...")
    dur_from_dates = (df["Fecha destino"] - df["Fecha origen"]).dt.total_seconds() / 60.0
    df["dur_min"] = dur_from_dates.fillna(0.0)
    
    # Limpiar duraciones negativas o muy altas (más de 24 horas)
    df["dur_min"] = df["dur_min"].where((df["dur_min"] >= 0) & (df["dur_min"] <= 1440), 0.0)
    
    # ============================
    # PASO 3: Limpieza de texto
    # ============================
    print("\nLimpiando texto...")
    
    def fix_mojibake(s: str) -> str:
        if pd.isna(s): 
            return ""
        s = str(s).strip()
        if any(ch in s for ch in ["Ã", "Â", "ð", "Ð"]):
            try: 
                s = s.encode("latin1").decode("utf-8")
            except Exception: 
                pass
        return " ".join(s.split())
    
    def normalize_key(s: str) -> str:
        s = fix_mojibake(str(s))
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return " ".join(s.upper().split())
    
    # Limpiar texto de usuarios y estaciones
    df["Usuario_clean"] = df["Usuario"].map(fix_mojibake)
    df["Origen_clean"] = df["Origen"].map(fix_mojibake)
    df["Destino_clean"] = df["Destino"].map(fix_mojibake)
    
    # Normalizar nombres para agrupar
    df["Usuario_key"] = df["Usuario"].map(normalize_key)
    
    # ============================
    # PASO 4: Crear features temporales
    # ============================
    print("\nCreando features temporales...")
    
    df["hora_salida"] = df["Fecha origen"].dt.hour.fillna(0).astype("int16")
    df["dia_semana"] = df["Fecha origen"].dt.dayofweek.fillna(0).astype("int8")
    df["mes"] = df["Fecha origen"].dt.month.fillna(1).astype("int8")
    df["semana"] = df["Fecha origen"].dt.to_period("W").astype(str)
    
    # ============================
    # PASO 5: Calcular métricas de usuario
    # ============================
    print("\nCalculando metricas de usuario...")
    
    def top1(s: pd.Series):
        vc = s.value_counts(dropna=True)
        return vc.index[0] if not vc.empty else "SIN DATOS"
    
    # Resumen por usuario
    resumen_usuarios = (
        df.groupby("Usuario_key", dropna=False)
          .agg(
              Usuario=("Usuario_clean", top1),
              viajes_totales=("Usuario_key", "size"),
              semanas_activas=("semana", pd.Series.nunique),
              origen_mas_frec=("Origen_clean", top1),
              destino_mas_frec=("Destino_clean", top1),
              hora_prom_salida=("hora_salida", "mean"),
              duracion_promedio_min=("dur_min", "mean"),
          )
          .reset_index()
    )
    
    # Viajes por semana
    den = resumen_usuarios["semanas_activas"].replace(0, 1)
    resumen_usuarios["viajes_por_semana"] = (resumen_usuarios["viajes_totales"] / den).round(2)
    
    print(f"     Usuarios unicos procesados: {len(resumen_usuarios)}")
    
    # ============================
    # PASO 6: Merge con estaciones
    # ============================
    print("\nIntegrando coordenadas de estaciones...")
    
    # Limpiar nombres de estaciones
    df_estaciones["station_name_clean"] = df_estaciones["station_name"].map(fix_mojibake)
    
    # Merge para origen
    df_merged = df.merge(
        df_estaciones[["station_name_clean", "station_lat", "station_lon"]],
        left_on="Origen_clean", 
        right_on="station_name_clean", 
        how="left"
    ).rename(columns={"station_lat": "origen_lat", "station_lon": "origen_lon"})
    
    # Merge para destino
    df_merged = df_merged.merge(
        df_estaciones[["station_name_clean", "station_lat", "station_lon"]],
        left_on="Destino_clean", 
        right_on="station_name_clean", 
        how="left"
    ).rename(columns={"station_lat": "destino_lat", "station_lon": "destino_lon"})
    
    # Merge con métricas de usuario
    df_merged = df_merged.merge(
        resumen_usuarios[["Usuario_key", "viajes_totales", "semanas_activas", 
                         "viajes_por_semana", "duracion_promedio_min"]],
        left_on="Usuario_key",
        right_on="Usuario_key",
        how="left"
    )
    
    print(f"     Registros con coordenadas: {df_merged['origen_lat'].notna().sum():,}")
    
    # ============================
    # PASO 7: Filtrar y limpiar datos finales
    # ============================
    print("\nFiltrando datos finales...")
    
    # Obtener estaciones válidas
    estaciones_validas = set(df_estaciones["station_name_clean"].astype(str).str.strip().unique())
    
    # Filtrar solo viajes con coordenadas válidas y destinos válidos
    mask_valid = (
        df_merged["origen_lat"].notna() & 
        df_merged["origen_lon"].notna() &
        df_merged["Destino_clean"].isin(estaciones_validas)
    )
    
    df_final = df_merged[mask_valid].copy()
    
    # Quitar destinos con muy pocos ejemplos (menos de 10)
    destino_counts = df_final["Destino_clean"].value_counts()
    destinos_validos = destino_counts[destino_counts >= 10].index
    df_final = df_final[df_final["Destino_clean"].isin(destinos_validos)].copy()
    
    print(f"     Registros finales validos: {len(df_final):,}")
    print(f"     Destinos unicos: {df_final['Destino_clean'].nunique()}")
    
    # ============================
    # PASO 8: Seleccionar features finales
    # ============================
    print("\nPreparando features finales...")
    
    # Features para el modelo
    features = [
        'origen_lat', 'origen_lon',
        'hora_salida', 'dia_semana', 'mes',
        'viajes_totales', 'semanas_activas', 'viajes_por_semana', 'duracion_promedio_min'
    ]
    
    # Rellenar valores faltantes
    df_final[features] = df_final[features].fillna(0)
    
    # Dataset final para el modelo
    dataset_modelo = df_final[features + ['Destino_clean']].copy()
    dataset_modelo.columns = features + ['destino']
    
    # ============================
    # PASO 9: Guardar dataset final
    # ============================
    print("\nGuardando dataset final...")
    
    OUTPUT_FILE = "dataset_modelo_stream.csv"
    dataset_modelo.to_csv(OUTPUT_FILE, index=False)
    
    print(f"     Dataset guardado en: {OUTPUT_FILE}")
    print(f"     Dimensiones: {dataset_modelo.shape[0]:,} filas x {dataset_modelo.shape[1]} columnas")
    
    # ============================
    # PASO 10: Estadísticas finales
    # ============================
    print("\nEstadisticas del dataset creado:")
    print(f"   - Total de registros: {len(dataset_modelo):,}")
    print(f"   - Destinos unicos: {dataset_modelo['destino'].nunique()}")
    print(f"   - Features utilizadas: {len(features)}")
    
    print(f"\nTop 10 destinos mas frecuentes:")
    top_destinos = dataset_modelo['destino'].value_counts().head(10)
    for i, (destino, count) in enumerate(top_destinos.items(), 1):
        print(f"   {i:2d}. {destino}: {count:,} viajes")
    
    print(f"\nDistribucion por hora de salida:")
    hora_dist = dataset_modelo['hora_salida'].value_counts().sort_index()
    for hora, count in hora_dist.head(10).items():
        print(f"   {hora:2d}:00 - {count:,} viajes")
    
    print(f"\nDistribucion por dia de la semana:")
    dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
    dia_dist = dataset_modelo['dia_semana'].value_counts().sort_index()
    for dia_num, count in dia_dist.items():
        print(f"   {dias[dia_num]}: {count:,} viajes")
    
    print(f"\nPROCESO COMPLETADO EXITOSAMENTE!")
    print(f"   El archivo {OUTPUT_FILE} esta listo para entrenar el modelo.")
    
    return dataset_modelo

if __name__ == "__main__":
    try:
        dataset = main()
        print(f"\nDataset creado exitosamente!")
        print(f"   Archivo: dataset_modelo_stream.csv")
        print(f"   Registros: {len(dataset):,}")
    except Exception as e:
        print(f"\nError durante la ejecucion: {e}")
        import traceback
        traceback.print_exc()
