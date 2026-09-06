#!/usr/bin/env python3
"""
Script para identificar la sonda más activa desde un CSV previo,
y luego descargar su historial continuo de 30 días usando la API de RIPE Atlas.
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta, timezone
from ripe.atlas.cousteau import AtlasResultsRequest
import sys

# ==========================================
# PARSING DE ARGUMENTOS
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extraer historial continuo de UNA SOLA SONDA desde RIPE Atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --csv-file ripe_atlas_ping_208230991.csv --measurement-id 208230991
  %(prog)s --csv-file ripe_atlas_ping_208230991.csv --measurement-id 208230991 --days 30
  %(prog)s --csv-file ripe_atlas_ping_208230991.csv --measurement-id 208230991 --days 14 --output my_data.csv
        """
    )
    
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Archivo CSV previo generado por extract_ripe_metrics.py"
    )
    
    parser.add_argument(
        "--measurement-id", "-m",
        type=int,
        required=True,
        help="ID de la medición de RIPE Atlas (msm_id)"
    )
    
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Días hacia atrás para consultar (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Archivo CSV de salida (default: historial_continuo_probe_<id>.csv)"
    )
    
    parser.add_argument(
        "--max-rtt",
        type=float,
        default=500.0,
        help="Filtro de outlier: RTT máximo en ms (default: %(default)s)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar información detallada"
    )
    
    return parser.parse_args()

# ==========================================
# ENCONTRAR MEJOR SONDA
# ==========================================
def encontrar_mejor_sonda(archivo_csv):
    """
    Analiza el CSV previo para encontrar la sonda con más mediciones.
    """
    print(f"🔍 Analizando {archivo_csv} para encontrar la sonda más activa...")
    try:
        df = pd.read_csv(archivo_csv)
        
        if 'probe_id' not in df.columns:
            print("❌ El archivo CSV no contiene la columna 'probe_id'")
            print(f"   Columnas disponibles: {', '.join(df.columns)}")
            sys.exit(1)
        
        # Contar frecuencias de cada probe_id
        probe_counts = df['probe_id'].value_counts()
        
        if probe_counts.empty:
            print("❌ No se encontraron probe_ids en el CSV.")
            sys.exit(1)
            
        mejor_probe = probe_counts.idxmax()
        cantidad = probe_counts.max()
        
        print(f"✅ Sonda identificada: Probe ID {mejor_probe}")
        print(f"   Total de mediciones de esta sonda en el CSV: {cantidad:,}")
        
        return int(mejor_probe)
        
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo {archivo_csv}.")
        print("   Ejecuta primero el script de extracción multi-sonda para generar este CSV.")
        sys.exit(1)
    except Exception as e:
        print(f" Error al procesar el CSV: {e}")
        sys.exit(1)

# ==========================================
# DESCARGAR HISTORIAL
# ==========================================
def descargar_historial_sonda(msm_id, probe_id, dias_atras):
    """
    Usa AtlasResultsRequest para descargar datos históricos de UNA sola sonda.
    """
    print(f"\n📥 Solicitando historial de {dias_atras} días para Probe ID {probe_id}...")
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=dias_atras)
    
    kwargs = {
        "msm_id": msm_id,
        "start": start_time,
        "stop": end_time,
        "probe_ids": [probe_id]
    }
    
    print(f"   Rango: {start_time.strftime('%Y-%m-%d')} a {end_time.strftime('%Y-%m-%d')}")
    print(f"   Measurement ID: {msm_id}")
    
    is_success, results = AtlasResultsRequest(**kwargs).create()
    
    if not is_success:
        print(f"❌ Error en la solicitud: {results}")
        return None
        
    print(f"   ✅ {len(results)} paquetes de medición descargados del servidor.")
    return results

# ==========================================
# PROCESAR Y VALIDAR
# ==========================================
def procesar_y_validar(results, probe_id, msm_id, max_rtt, output_file):
    """
    Parsea los resultados, calcula métricas de serie temporal y guarda en CSV.
    """
    print("\n⚙️  Procesando y validando continuidad temporal...")
    
    parsed_data = []
    target_addr = None
    
    for res in results:
        timestamp = res.get('timestamp')
        if target_addr is None:
            target_addr = res.get('dst_addr')
            
        for packet in res.get('result', []):
            if 'rtt' in packet and packet['rtt'] is not None:
                parsed_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                    'rtt_ms': float(packet['rtt']),
                    'probe_id': probe_id,
                    'asn': res.get('asn', 'Unknown')
                })
                
    if not parsed_data:
        print("❌ No se extrajeron datos de RTT válidos.")
        return None
        
    df = pd.DataFrame(parsed_data)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Filtrar outliers
    df = df[df['rtt_ms'] < max_rtt].copy()
    
    # Calcular Jitter Temporal
    df = df.sort_values(['probe_id', 'timestamp'])
    df['jitter_ms'] = df.groupby('probe_id')['rtt_ms'].diff().abs()
    
    # Guardar en archivo
    df.to_csv(output_file, index=True)
    print(f"💾 Datos guardados en: {output_file}")
    
    # ==========================================
    # VALIDACIÓN DE SERIE TEMPORAL
    # ==========================================
    print("\n🔬 Validando propiedades de Serie Temporal (Ground Truth):")
    print(f"   Target: {target_addr} | Probe: {probe_id} | MSM: {msm_id}")
    print(f"   ASN: {df['asn'].iloc[0] if len(df) > 0 else 'Unknown'}")
    print(f"   Muestras totales: {len(df):,}")
    
    if len(df) > 0:
        print(f"   Período: {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Mediana RTT: {df['rtt_ms'].median():.2f} ms")
        print(f"   Percentil 95 RTT: {df['rtt_ms'].quantile(0.95):.2f} ms")
        print(f"   Mediana Jitter: {df['jitter_ms'].median():.2f} ms")
        
        # 1. Autocorrelación (Lag-1)
        if len(df) > 1:
            autocorr = df['rtt_ms'].autocorr(lag=1)
            print(f"   Autocorrelación (Lag-1): {autocorr:.4f} {'✅ (Buena continuidad)' if autocorr > 0.1 else '⚠️  (Baja continuidad)'}")
        else:
            print("   Autocorrelación: N/A (insuficientes muestras)")
        
        # 2. Variación Diurna
        df['hour'] = df.index.hour
        df['is_peak'] = df['hour'].apply(lambda x: 1 if (18 <= x <= 23) or (8 <= x <= 10) else 0)
        
        median_offpeak = df[df['is_peak'] == 0]['rtt_ms'].median()
        median_peak = df[df['is_peak'] == 1]['rtt_ms'].median()
        
        if pd.notna(median_offpeak) and median_offpeak > 0 and pd.notna(median_peak):
            diurnal_var = ((median_peak - median_offpeak) / median_offpeak) * 100
            print(f"   Variación Diurna: {diurnal_var:.1f}% (Off-peak: {median_offpeak:.1f}ms, Peak: {median_peak:.1f}ms)")
        else:
            print("   Variación Diurna: No calculable (datos insuficientes o todos en mismo período)")
    else:
        print("   ⚠️  No hay datos suficientes para calcular estadísticas")
        
    return df

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    args = parse_arguments()
    
    print("="*70)
    print("EXTRACCIÓN DE HISTORIAL CONTINUO DE UNA SOLA SONDA (RIPE Atlas)")
    print("="*70)
    print(f"\n⚙️  Configuración:")
    print(f"   Archivo CSV: {args.csv_file}")
    print(f"   Measurement ID: {args.measurement_id}")
    print(f"   Días hacia atrás: {args.days}")
    print(f"   Filtro max RTT: {args.max_rtt} ms")
    if args.output:
        print(f"   Archivo de salida: {args.output}")
    
    # Paso 1: Encontrar la mejor sonda
    mejor_probe = encontrar_mejor_sonda(args.csv_file)
    
    # Paso 2: Descargar su historial
    resultados = descargar_historial_sonda(args.measurement_id, mejor_probe, args.days)
    
    # Paso 3: Procesar, validar y guardar
    if resultados:
        # Determinar nombre de archivo de salida
        output_file = args.output
        if output_file is None:
            output_file = f"historial_continuo_probe_{mejor_probe}.csv"
        
        df_final = procesar_y_validar(
            resultados, 
            mejor_probe, 
            args.measurement_id,
            args.max_rtt,
            output_file
        )
        
        if df_final is not None:
            print("\n✅ ¡Proceso completado! Ahora tienes un Ground Truth válido para series temporales.")
            print(f"   Siguiente paso: Usa este CSV para entrenar tu modelo generativo.")
        else:
            print("\n❌ Error al procesar los datos.")
            sys.exit(1)
    else:
        print("\n❌ No se pudieron descargar los resultados.")
        sys.exit(1)

if __name__ == "__main__":
    main()
