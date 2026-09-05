#!/usr/bin/env python3
"""
Script para extraer datos de latencia RTT de RIPE Atlas
para validación empírica del modelo generativo de latencia BGP.
"""

from ripe.atlas.cousteau import AtlasResultsRequest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ==========================================
# CONFIGURACIÓN
# ==========================================
TARGET_IP = "8.8.8.8"  # Google DNS (tiene buena cobertura global)
DAYS_BACK = 30         # Días hacia atrás para extraer datos

def download_results(measurement_id, days_back=DAYS_BACK):
    """
    Descarga los resultados de una medición existente
    """
    print(f"\n📥 Descargando resultados de la medición {measurement_id}...")
    
    # Calcular rango de tiempo (usando timezone.utc para evitar DeprecationWarning)
    end_time = int(datetime.now(timezone.utc).timestamp())
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    
    print(f"   Consultando desde {datetime.fromtimestamp(start_time, tz=timezone.utc).strftime('%Y-%m-%d')} hasta ahora...")
    
    # CORRECCIÓN: El parámetro se llama 'msm_id', no 'msms'
    is_success, results = AtlasResultsRequest(
        msm_id=measurement_id,
        start=start_time,
        stop=end_time
    ).create()
    
    if not is_success:
        print(f"❌ Error al descargar resultados: {results}")
        return None
    
    print(f"   Procesando resultados...")
    
    all_results = []
    # Parsear resultados
    for result_data in results:
        try:
            probe_id = result_data.get("prb_id")
            asn = result_data.get("asn", "Unknown")
            timestamp = result_data.get("timestamp")
            
            # Extraer RTT de cada paquete (la clave es "result" que contiene una lista)
            for packet in result_data.get("result", []):
                if "rtt" in packet and packet["rtt"] is not None:
                    all_results.append({
                        'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        'rtt_ms': float(packet["rtt"]),
                        'probe_id': probe_id,
                        'asn': asn
                    })
        except Exception:
            # Ignorar errores de parseo en paquetes individuales malformados
            continue
    
    print(f"   ✅ {len(all_results)} muestras de RTT extraídas")
    return all_results

def analyze_and_save(results, output_file="ripe_atlas_empirical_30days.csv"):
    """
    Analiza los resultados y los guarda en CSV
    """
    if not results:
        print("❌ No hay resultados para guardar")
        return None
    
    # Convertir a DataFrame
    df = pd.DataFrame(results)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Filtrar outliers extremos (> 500ms) que suelen ser errores de medición
    df = df[df['rtt_ms'] < 500].copy()
    
    print(f"\n📊 Estadísticas básicas de latencia RTT:")
    print(f"   Total de muestras: {len(df)}")
    print(f"   Mediana: {df['rtt_ms'].median():.2f} ms")
    print(f"   Percentil 95: {df['rtt_ms'].quantile(0.95):.2f} ms")
    print(f"   Máximo: {df['rtt_ms'].max():.2f} ms")
    print(f"   Mínimo: {df['rtt_ms'].min():.2f} ms")
    print(f"   Desviación estándar: {df['rtt_ms'].std():.2f} ms")
    
    # Guardar en CSV
    df.to_csv(output_file, index=True)
    print(f"\n💾 Datos guardados en '{output_file}'")
    
    return df

def calculate_empirical_metrics(df):
    """
    Calcula las métricas empíricas para validar el modelo generativo
    """
    print("\n🔬 Calculando métricas empíricas para validación...")
    
    rtt = df['rtt_ms'].dropna()
    
    # 1. Distribución y Percentiles
    metrics = {
        'median': np.median(rtt),
        'p95': np.percentile(rtt, 95),
        'p99': np.percentile(rtt, 99),
        'max': np.max(rtt),
        'min': np.min(rtt),
        'std': np.std(rtt),
        'mean': np.mean(rtt)
    }
    
    # 2. Variación Diurna
    df['hour'] = df.index.hour
    df['is_peak'] = df['hour'].apply(lambda x: 1 if 18 <= x <= 23 or 8 <= x <= 10 else 0)
    
    median_offpeak = df[df['is_peak'] == 0]['rtt_ms'].median()
    median_peak = df[df['is_peak'] == 1]['rtt_ms'].median()
    metrics['diurnal_var_pct'] = ((median_peak - median_offpeak) / median_offpeak) * 100 if median_offpeak > 0 else 0
    
    # 3. Autocorrelación (Lag-1)
    # Muestreamos a 1 medición por minuto por sonda para evitar sesgo de alta frecuencia
    df_resampled = df.groupby([pd.Grouper(freq='1min'), "probe_id"])["rtt_ms"].mean().reset_index()
    if len(df_resampled) > 1:
        df_pivot = df_resampled.pivot(index="timestamp", columns="probe_id", values="rtt_ms").mean(axis=1)
        metrics['autocorr_lag1'] = df_pivot.autocorr(lag=1)
    else:
        metrics['autocorr_lag1'] = 0.0
    
    print(f"\n--- MÉTRICAS EMPÍRICAS ---")
    print(f"Mediana RTT: {metrics['median']:.2f} ms")
    print(f"Percentil 95: {metrics['p95']:.2f} ms")
    print(f"Percentil 99: {metrics['p99']:.2f} ms")
    print(f"Variación Diurna: {metrics['diurnal_var_pct']:.1f}%")
    print(f"Autocorrelación (Lag-1): {metrics['autocorr_lag1']:.4f}")
    
    return metrics

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("="*70)
    print("RIPE Atlas - Extracción de Datos de Latencia para Validación Empírica")
    print("="*70)
    
    # ID de medición pública (ping a 8.8.8.8). 
    measurement_id = 208230991  
    
    if measurement_id:
        # Descargar resultados (últimos 30 días)
        results = download_results(measurement_id, days_back=30)
        
        if results:
            # Analizar y guardar
            df = analyze_and_save(results, output_file="ripe_atlas_empirical_30days.csv")
            
            if df is not None:
                # Calcular métricas empíricas
                empirical_metrics = calculate_empirical_metrics(df)
                
                print("\n✅ Proceso completado exitosamente!")
                print("💡 Usa estas métricas para validar tu modelo generativo:")
                print(f"   - Configura la mediana objetivo en: {empirical_metrics['median']:.2f} ms")
                print(f"   - Configura el percentil 95 objetivo en: {empirical_metrics['p95']:.2f} ms")
                print(f"   - Configura la variación diurna en: {empirical_metrics['diurnal_var_pct']:.1f}%")
        else:
            print("⚠️ No se pudieron descargar resultados. Verifica que el ID de medición sea público y tenga datos recientes.")
