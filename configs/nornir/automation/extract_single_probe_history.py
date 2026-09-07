#!/usr/bin/env python3
"""
Script para extraer historial continuo de UNA SOLA SONDA desde RIPE Atlas.
Agrega los 3 paquetes por ciclo calculando avg/stddev.
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
  %(prog)s --measurement-id 26304408 --probe-id 54061
  %(prog)s --measurement-id 26304408 --probe-id 54061 --days 30
  %(prog)s --measurement-id 26304408 --probe-id 54061 --days 90 --output my_data.csv
        """
    )
    
    parser.add_argument(
        "--measurement-id", "-m",
        type=int,
        required=True,
        help="ID de la medición de RIPE Atlas (msm_id)"
    )
    
    parser.add_argument(
        "--probe-id", "-p",
        type=int,
        required=True,
        help="ID de la sonda específica a consultar (probe_id)"
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
    
    print(f"   ✅ {len(results)} mediciones descargadas del servidor.")
    return results


# ==========================================
# PROCESAR Y AGREGAR POR CICLO
# ==========================================
def procesar_y_agregar_por_ciclo(results, probe_id, msm_id):
    """
    Procesa los resultados y agrega los 3 paquetes por ciclo.
    Retorna un DataFrame con una fila por ciclo (no por paquete).
    """
    print("\n🔍 Procesando resultados y agregando por ciclo...")
    
    all_data = []
    target_addr = None
    
    for res in results:
        try:
            timestamp = datetime.fromtimestamp(res.get('timestamp'), tz=timezone.utc)
            asn = res.get('asn', 'Unknown')
            
            if target_addr is None:
                target_addr = res.get('dst_addr')
            
            # Extraer los 3 RTT del ciclo
            rtts = []
            for packet in res.get('result', []):
                if 'rtt' in packet and packet['rtt'] is not None:
                    rtts.append(float(packet['rtt']))
            
            if len(rtts) >= 1:
                # Calcular avg y stddev de los paquetes del ciclo
                avg_rtt = np.mean(rtts)
                std_rtt = np.std(rtts) if len(rtts) > 1 else 0.0
                
                all_data.append({
                    'timestamp': timestamp,
                    'cycle_number': None,  # Se asignará después
                    'probe_id': probe_id,
                    'asn': asn,
                    'target': target_addr,
                    'rtt_avg_ms': avg_rtt,
                    'rtt_std_ms': std_rtt,
                    'rtt_min_ms': min(rtts),
                    'rtt_max_ms': max(rtts),
                    'packets_count': len(rtts)
                })
        except Exception as e:
            if args.verbose:
                print(f"   ⚠️ Error procesando resultado: {e}")
            continue
    
    if not all_data:
        print("   ❌ No se extrajeron datos válidos")
        return None, None
    
    # Crear DataFrame
    df = pd.DataFrame(all_data)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Asignar cycle_number secuencial
    df['cycle_number'] = range(1, len(df) + 1)
    
    # Reordenar columnas
    df = df[['cycle_number', 'probe_id', 'target', 'asn', 
             'rtt_avg_ms', 'rtt_std_ms', 'rtt_min_ms', 'rtt_max_ms', 'packets_count']]
    
    print(f"   ✅ {len(df)} ciclos procesados")
    
    return df, target_addr


# ==========================================
# VALIDAR Y GUARDAR
# ==========================================
def validar_y_guardar(df, target_addr, probe_id, msm_id, max_rtt, output_file):
    """
    Filtra outliers, calcula métricas de serie temporal y guarda en CSV.
    """
    print("\n⚙️  Validando y calculando métricas de serie temporal...")
    
    # Filtrar outliers
    df_filtered = df[df['rtt_avg_ms'] < max_rtt].copy()
    
    if output_file is None:
        output_file = f"historial_continuo_probe_{probe_id}.csv"
    
    df_filtered.to_csv(output_file, index=True)
    print(f"💾 Datos guardados en: {output_file}")
    
    # ==========================================
    # VALIDACIÓN DE SERIE TEMPORAL
    # ==========================================
    print("\n🔬 Validando propiedades de Serie Temporal (Ground Truth):")
    print(f"   Target: {target_addr} | Probe: {probe_id} | MSM: {msm_id}")
    print(f"   ASN: {df_filtered['asn'].iloc[0] if len(df_filtered) > 0 else 'Unknown'}")
    print(f"   Muestras totales (ciclos): {len(df_filtered):,}")
    
    if len(df_filtered) > 0:
        print(f"   Período: {df_filtered.index[0].strftime('%Y-%m-%d %H:%M')} → {df_filtered.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Mediana RTT (avg): {df_filtered['rtt_avg_ms'].median():.2f} ms")
        print(f"   Percentil 95 RTT (avg): {df_filtered['rtt_avg_ms'].quantile(0.95):.2f} ms")
        print(f"   Mediana RTT std: {df_filtered['rtt_std_ms'].median():.2f} ms")
        print(f"   RTT min: {df_filtered['rtt_min_ms'].min():.2f} ms")
        print(f"   RTT max: {df_filtered['rtt_max_ms'].max():.2f} ms")
        
        # Calcular Jitter Temporal (variación entre ciclos consecutivos)
        df_filtered = df_filtered.sort_values(['probe_id', 'timestamp'])
        df_filtered['jitter_ms'] = df_filtered.groupby('probe_id')['rtt_avg_ms'].diff().abs()
        print(f"   Mediana Jitter: {df_filtered['jitter_ms'].median():.2f} ms")
        
        # 1. Autocorrelación (Lag-1)
        if len(df_filtered) > 1:
            autocorr = df_filtered['rtt_avg_ms'].autocorr(lag=1)
            print(f"   Autocorrelación (Lag-1): {autocorr:.4f} {'✅ (Buena continuidad)' if autocorr > 0.1 else '⚠️  (Baja continuidad)'}")
        else:
            print("   Autocorrelación: N/A (insuficientes muestras)")
        
        # 2. Variación Diurna
        df_filtered['hour'] = df_filtered.index.hour
        df_filtered['is_peak'] = df_filtered['hour'].apply(lambda x: 1 if (18 <= x <= 23) or (8 <= x <= 10) else 0)
        
        median_offpeak = df_filtered[df_filtered['is_peak'] == 0]['rtt_avg_ms'].median()
        median_peak = df_filtered[df_filtered['is_peak'] == 1]['rtt_avg_ms'].median()
        
        if pd.notna(median_offpeak) and median_offpeak > 0 and pd.notna(median_peak):
            diurnal_var = ((median_peak - median_offpeak) / median_offpeak) * 100
            print(f"   Variación Diurna: {diurnal_var:.1f}% (Off-peak: {median_offpeak:.1f}ms, Peak: {median_peak:.1f}ms)")
        else:
            print("   Variación Diurna: No calculable (datos insuficientes o todos en mismo período)")
    else:
        print("   ⚠️  No hay datos suficientes para calcular estadísticas")
    
    return df_filtered


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    global args
    args = parse_arguments()
    
    print("="*70)
    print("EXTRACCIÓN DE HISTORIAL CONTINUO DE UNA SOLA SONDA (RIPE Atlas)")
    print("="*70)
    print(f"\n⚙️  Configuración:")
    print(f"   Measurement ID: {args.measurement_id}")
    print(f"   Probe ID: {args.probe_id}")
    print(f"   Días hacia atrás: {args.days}")
    print(f"   Filtro max RTT: {args.max_rtt} ms")
    if args.output:
        print(f"   Archivo de salida: {args.output}")
    
    # Paso 1: Descargar historial
    resultados = descargar_historial_sonda(args.measurement_id, args.probe_id, args.days)
    
    if not resultados:
        print("\n No se pudieron descargar los resultados.")
        sys.exit(1)
    
    # Paso 2: Procesar y agregar por ciclo
    df, target_addr = procesar_y_agregar_por_ciclo(
        resultados, 
        args.probe_id, 
        args.measurement_id
    )
    
    if df is None:
        print("\n❌ Error al procesar los datos.")
        sys.exit(1)
    
    # Paso 3: Validar y guardar
    df_final = validar_y_guardar(
        df, 
        target_addr, 
        args.probe_id, 
        args.measurement_id,
        args.max_rtt,
        args.output
    )
    
    if df_final is not None:
        print("\n✅ ¡Proceso completado!")
        print(f"   Ahora tienes un Ground Truth válido para series temporales.")
        print(f"   Siguiente paso: Usa este CSV para entrenar tu modelo generativo.")
    else:
        print("\n❌ Error al validar los datos.")
        sys.exit(1)


if __name__ == "__main__":
    main()
