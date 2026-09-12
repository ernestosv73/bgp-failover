#!/usr/bin/env python3
"""
Script para extraer historial continuo de UNA SOLA SONDA desde RIPE Atlas.
Agrega los 3 paquetes por ciclo calculando avg/stddev.
Calcula 3 tipos de jitter: intra-ciclo, inter-ciclo y RFC 3550.
Incluye resolución de ASN y country_code desde metadata de la sonda.
Calcula variación diurna usando la zona horaria local de la sonda.
Usa autocorrelación de Spearman (robusta a outliers).
"""
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from scipy.stats import spearmanr
from ripe.atlas.cousteau import AtlasResultsRequest, Probe
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
# OBTENER METADATA DE LA SONDA
# ==========================================
def get_probe_metadata(probe_id):
    """
    Obtiene metadata de una sonda (ASN, country_code) usando la API de RIPE Atlas.
    """
    print(f"\n🔍 Obteniendo metadata de Probe ID {probe_id}...")
    
    try:
        probe = Probe(id=probe_id)
        metadata = {
            'asn_v4': probe.asn_v4 if hasattr(probe, 'asn_v4') else 'Unknown',
            'asn_v6': probe.asn_v6 if hasattr(probe, 'asn_v6') else 'Unknown',
            'country_code': probe.country_code if hasattr(probe, 'country_code') else 'Unknown',
            'is_anchor': probe.is_anchor if hasattr(probe, 'is_anchor') else False,
            'address_v4': probe.address_v4 if hasattr(probe, 'address_v4') else None,
        }
        
        print(f"   ✅ ASN: {metadata['asn_v4']}, Country: {metadata['country_code']}")
        return metadata
        
    except Exception as e:
        print(f"   ⚠️ Error obteniendo metadata: {e}")
        return {
            'asn_v4': 'Unknown',
            'asn_v6': 'Unknown',
            'country_code': 'Unknown',
            'is_anchor': False,
            'address_v4': None,
        }


# ==========================================
# OBTENER ZONA HORARIA DE LA SONDA
# ==========================================
def get_probe_timezone(probe_id, country_code):
    """
    Obtiene la zona horaria de la sonda basada en su country_code.
    Retorna un objeto ZoneInfo o UTC por defecto.
    """
    # Mapeo de país a zona horaria principal
    country_to_tz = {
        'DE': 'Europe/Berlin',
        'US': 'America/New_York',
        'BR': 'America/Sao_Paulo',
        'UY': 'America/Montevideo',
        'NL': 'Europe/Amsterdam',
        'FR': 'Europe/Paris',
        'GB': 'Europe/London',
        'JP': 'Asia/Tokyo',
        'AU': 'Australia/Sydney',
        'ES': 'Europe/Madrid',
        'IT': 'Europe/Rome',
        'CA': 'America/Toronto',
        'MX': 'America/Mexico_City',
        'AR': 'America/Argentina/Buenos_Aires',
        'CL': 'America/Santiago',
        'CO': 'America/Bogota',
        'PE': 'America/Lima',
        'RU': 'Europe/Moscow',
        'CN': 'Asia/Shanghai',
        'IN': 'Asia/Kolkata',
        'KR': 'Asia/Seoul',
        'SG': 'Asia/Singapore',
        'PL': 'Europe/Warsaw',
        'SE': 'Europe/Stockholm',
        'NO': 'Europe/Oslo',
        'FI': 'Europe/Helsinki',
        'DK': 'Europe/Copenhagen',
        'BE': 'Europe/Brussels',
        'AT': 'Europe/Vienna',
        'CH': 'Europe/Zurich',
        'PT': 'Europe/Lisbon',
        'GR': 'Europe/Athens',
        'TR': 'Europe/Istanbul',
        'IL': 'Asia/Jerusalem',
        'ZA': 'Africa/Johannesburg',
        'EG': 'Africa/Cairo',
        'NG': 'Africa/Lagos',
        'KE': 'Africa/Nairobi',
        'NZ': 'Pacific/Auckland',
    }
    
    if country_code and country_code in country_to_tz:
        tz_name = country_to_tz[country_code]
        print(f"   🌍 Zona horaria detectada para Probe {probe_id} ({country_code}): {tz_name}")
        return ZoneInfo(tz_name)
    else:
        print(f"   ⚠️ Zona horaria no disponible para país '{country_code}', usando UTC")
        return ZoneInfo('UTC')


# ==========================================
# CALCULAR AUTOCORRELACIÓN DE SPEARMAN
# ==========================================
def calcular_autocorrelacion_spearman(series, lag=1):
    """
    Calcula la autocorrelación usando el coeficiente de Spearman.
    Robusta a outliers y no asume distribución normal.
    
    Args:
        series: Serie temporal de pandas
        lag: Retraso temporal (default: 1)
    
    Returns:
        coeficiente de correlación de Spearman (float)
    """
    # Crear serie desplazada
    series_shifted = series.shift(lag)
    
    # Eliminar valores NaN
    valid_pairs = pd.concat([series, series_shifted], axis=1).dropna()
    
    if len(valid_pairs) < 10:  # Mínimo de muestras para cálculo significativo
        return float('nan')
    
    # Calcular correlación de Spearman
    coeficiente, p_value = spearmanr(valid_pairs.iloc[:, 0], valid_pairs.iloc[:, 1])
    
    return coeficiente


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
# PROCESAR, AGREGAR POR CICLO Y CALCULAR JITTERS
# ==========================================
def procesar_y_calcular_jitters(results, probe_id, msm_id, probe_metadata):
    """
    Procesa los resultados, agrega los 3 paquetes por ciclo y calcula:
    1. Jitter Intra-Ciclo (std dev de los paquetes del mismo ciclo)
    2. Jitter Inter-Ciclo (variación entre promedios de ciclos consecutivos)
    3. Jitter RFC 3550 (IPDV - Inter-Packet Delay Variation)
    """
    print("\n🔍 Procesando resultados, agregando por ciclo y calculando jitters...")
    
    all_data = []
    target_addr = None
    prev_rtt_avg = None  # Para calcular jitter inter-ciclo
    
    for res in results:
        try:
            timestamp = datetime.fromtimestamp(res.get('timestamp'), tz=timezone.utc)
            
            if target_addr is None:
                target_addr = res.get('dst_addr')
            
            # Extraer los RTTs del ciclo
            rtts = []
            for packet in res.get('result', []):
                if 'rtt' in packet and packet['rtt'] is not None:
                    rtts.append(float(packet['rtt']))
            
            if len(rtts) >= 1:
                # Métricas básicas del ciclo
                avg_rtt = np.mean(rtts)
                std_rtt = np.std(rtts) if len(rtts) > 1 else 0.0
                
                # 1. Jitter Intra-Ciclo (std dev de los paquetes del ciclo)
                jitter_intra = std_rtt
                
                # 2. Jitter Inter-Ciclo (variación entre ciclos consecutivos)
                jitter_inter = abs(avg_rtt - prev_rtt_avg) if prev_rtt_avg is not None else 0.0
                
                # Actualizar para el próximo ciclo
                prev_rtt_avg = avg_rtt
                
                all_data.append({
                    'timestamp': timestamp,
                    'cycle_number': None,  # Se asignará después
                    'probe_id': probe_id,
                    'asn': probe_metadata['asn_v4'],
                    'country_code': probe_metadata['country_code'],
                    'target': target_addr,
                    'rtt_avg_ms': avg_rtt,
                    'rtt_std_ms': std_rtt,
                    'rtt_min_ms': min(rtts),
                    'rtt_max_ms': max(rtts),
                    'packets_count': len(rtts),
                    'jitter_intra_ms': jitter_intra,
                    'jitter_inter_ms': jitter_inter,
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
    
    # 3. Jitter RFC 3550 (IPDV - variación en el delay entre ciclos consecutivos)
    # Se calcula sobre toda la serie temporal ordenada
    df['jitter_rfc3550_ms'] = df['rtt_avg_ms'].diff().abs()
    
    # Reordenar columnas
    df = df[['cycle_number', 'probe_id', 'asn', 'country_code', 'target',
             'rtt_avg_ms', 'rtt_std_ms', 'rtt_min_ms', 'rtt_max_ms', 'packets_count',
             'jitter_intra_ms', 'jitter_inter_ms', 'jitter_rfc3550_ms']]
    
    print(f"   ✅ {len(df)} ciclos procesados")
    
    # Mostrar resumen de jitters
    print(f"\n📊 Resumen de Jitters:")
    print(f"   Jitter Intra-Ciclo (std dev intra-ciclo):")
    print(f"      Mediana: {df['jitter_intra_ms'].median():.3f} ms")
    print(f"      Percentil 95: {df['jitter_intra_ms'].quantile(0.95):.3f} ms")
    print(f"   Jitter Inter-Ciclo (variación entre ciclos):")
    print(f"      Mediana: {df['jitter_inter_ms'].median():.3f} ms")
    print(f"      Percentil 95: {df['jitter_inter_ms'].quantile(0.95):.3f} ms")
    print(f"   Jitter RFC 3550 (IPDV):")
    print(f"      Mediana: {df['jitter_rfc3550_ms'].median():.3f} ms")
    print(f"      Percentil 95: {df['jitter_rfc3550_ms'].quantile(0.95):.3f} ms")
    
    return df, target_addr


# ==========================================
# VALIDAR Y GUARDAR
# ==========================================
def validar_y_guardar(df, target_addr, probe_id, msm_id, probe_metadata, max_rtt, output_file):
    """
    Filtra outliers, calcula métricas de serie temporal y guarda en CSV.
    Incluye cálculo de variación diurna usando zona horaria local.
    Usa autocorrelación de Spearman (robusta a outliers).
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
    print(f"   Target: {target_addr}")
    print(f"   Probe: {probe_id} ({probe_metadata['country_code']}, ASN {probe_metadata['asn_v4']})")
    print(f"   MSM: {msm_id}")
    print(f"   Muestras totales (ciclos): {len(df_filtered):,}")
    
    if len(df_filtered) > 0:
        print(f"   Período: {df_filtered.index[0].strftime('%Y-%m-%d %H:%M')} → {df_filtered.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Mediana RTT (avg): {df_filtered['rtt_avg_ms'].median():.2f} ms")
        print(f"   Percentil 95 RTT (avg): {df_filtered['rtt_avg_ms'].quantile(0.95):.2f} ms")
        print(f"   Mediana RTT std: {df_filtered['rtt_std_ms'].median():.2f} ms")
        print(f"   RTT min: {df_filtered['rtt_min_ms'].min():.2f} ms")
        print(f"   RTT max: {df_filtered['rtt_max_ms'].max():.2f} ms")
        
        # 1. Autocorrelación de Spearman (Lag-1) - ROBUSTA A OUTLIERS
        if len(df_filtered) > 10:
            autocorr_spearman = calcular_autocorrelacion_spearman(df_filtered['rtt_avg_ms'], lag=1)
            
            # También calcular Pearson para comparación
            autocorr_pearson = df_filtered['rtt_avg_ms'].autocorr(lag=1)
            
            print(f"\n   📊 Autocorrelación Temporal:")
            print(f"      Spearman (Lag-1): {autocorr_spearman:.4f} {'✅ (Buena continuidad)' if autocorr_spearman > 0.3 else '⚠️  (Baja continuidad)'}")
            print(f"      Pearson (Lag-1): {autocorr_pearson:.4f} {'✅' if autocorr_pearson > 0.3 else '⚠️'}")
            
            # Explicar la diferencia si es significativa
            diff = abs(autocorr_spearman - autocorr_pearson)
            if diff > 0.2:
                print(f"      ℹ️  Diferencia significativa ({diff:.3f}): Spearman es más confiable por presencia de outliers")
        else:
            print("   Autocorrelación: N/A (insuficientes muestras)")
        
        # 2. Variación Diurna (con zona horaria local)
        probe_tz = get_probe_timezone(probe_id, probe_metadata['country_code'])
        
        # Convertir timestamps UTC a hora local
        df_filtered['local_time'] = df_filtered.index.tz_convert(probe_tz)
        df_filtered['local_hour'] = df_filtered['local_time'].dt.hour
        
        # Definir peak/off-peak en hora LOCAL
        # Peak: 07:00-09:59 y 19:00-23:59 (hora local)
        # Off-peak: 00:00-06:59 y 10:00-18:59 (hora local)
        df_filtered['is_peak'] = df_filtered['local_hour'].apply(
            lambda x: 1 if (7 <= x <= 9) or (19 <= x <= 23) else 0
        )
        
        median_offpeak = df_filtered[df_filtered['is_peak'] == 0]['rtt_avg_ms'].median()
        median_peak = df_filtered[df_filtered['is_peak'] == 1]['rtt_avg_ms'].median()
        
        print(f"\n    Variación Diurna (hora local {probe_tz}):")
        print(f"      Peak hours (07-10h, 19-24h local): {median_peak:.2f} ms")
        print(f"      Off-peak hours (00-07h, 10-19h local): {median_offpeak:.2f} ms")
        
        if pd.notna(median_offpeak) and median_offpeak > 0 and pd.notna(median_peak):
            diurnal_var = ((median_peak - median_offpeak) / median_offpeak) * 100
            print(f"      Variación Diurna: {diurnal_var:.1f}%")
            
            if diurnal_var > 5:
                print(f"      ✅ Patrón diurno positivo detectado (mayor latencia en horas pico)")
            elif diurnal_var < -5:
                print(f"      ️  Patrón diurno negativo (menor latencia en horas pico - posible artefacto)")
            else:
                print(f"      ℹ️  Variación diurna mínima (path estable)")
        else:
            print("      ⚠️  Variación Diurna: No calculable (datos insuficientes)")
        
        # 3. Estadísticas detalladas de Jitters
        print(f"\n Estadísticas detalladas de Jitters:")
        
        # Jitter Intra-Ciclo
        print(f"   Jitter Intra-Ciclo (std dev intra-ciclo):")
        print(f"      Mediana: {df_filtered['jitter_intra_ms'].median():.3f} ms")
        print(f"      Percentil 95: {df_filtered['jitter_intra_ms'].quantile(0.95):.3f} ms")
        print(f"      Máximo: {df_filtered['jitter_intra_ms'].max():.3f} ms")
        
        # Jitter Inter-Ciclo
        print(f"   Jitter Inter-Ciclo (variación entre ciclos):")
        print(f"      Mediana: {df_filtered['jitter_inter_ms'].median():.3f} ms")
        print(f"      Percentil 95: {df_filtered['jitter_inter_ms'].quantile(0.95):.3f} ms")
        print(f"      Máximo: {df_filtered['jitter_inter_ms'].max():.3f} ms")
        
        # Jitter RFC 3550
        print(f"   Jitter RFC 3550 (IPDV):")
        print(f"      Mediana: {df_filtered['jitter_rfc3550_ms'].median():.3f} ms")
        print(f"      Percentil 95: {df_filtered['jitter_rfc3550_ms'].quantile(0.95):.3f} ms")
        print(f"      Máximo: {df_filtered['jitter_rfc3550_ms'].max():.3f} ms")
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
    
    # Paso 1: Obtener metadata de la sonda
    probe_metadata = get_probe_metadata(args.probe_id)
    
    # Paso 2: Descargar historial
    resultados = descargar_historial_sonda(args.measurement_id, args.probe_id, args.days)
    
    if not resultados:
        print("\n❌ No se pudieron descargar los resultados.")
        sys.exit(1)
    
    # Paso 3: Procesar, agregar por ciclo y calcular jitters
    df, target_addr = procesar_y_calcular_jitters(
        resultados, 
        args.probe_id, 
        args.measurement_id,
        probe_metadata
    )
    
    if df is None:
        print("\n❌ Error al procesar los datos.")
        sys.exit(1)
    
    # Paso 4: Validar y guardar
    df_final = validar_y_guardar(
        df, 
        target_addr, 
        args.probe_id, 
        args.measurement_id,
        probe_metadata,
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
