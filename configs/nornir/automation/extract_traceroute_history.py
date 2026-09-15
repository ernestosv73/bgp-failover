#!/usr/bin/env python3
"""
Script para extraer historial de TRACEROUTE de UNA SOLA SONDA desde RIPE Atlas.
Extrae métricas salto por salto (hop-by-hop) con detección adaptativa de anomalías IQR.
"""
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime, timedelta, timezone
from scipy.stats import spearmanr
from ripe.atlas.cousteau import AtlasResultsRequest, Probe


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extraer historial de TRACEROUTE de UNA SOLA SONDA desde RIPE Atlas "
                    "con detección adaptativa de anomalías por salto (IQR Tukey k=1.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --measurement-id 154280895 --probe-id 1014401 --days 7
  %(prog)s --measurement-id 154280895 --probe-id 1014401 \\
           --start-time "2026-09-13 00:00" --stop-time "2026-09-15 00:00"
  %(prog)s --measurement-id 154280895 --probe-id 1014401 --days 7 --iqr-k 3.0
        """
    )
    
    parser.add_argument("--measurement-id", "-m", type=int, required=True)
    parser.add_argument("--probe-id", "-p", type=int, required=True)
    parser.add_argument("--days", "-d", type=int, help="Días hacia atrás desde ahora")
    parser.add_argument("--start-time", type=str, help="Fecha/hora inicio")
    parser.add_argument("--stop-time", type=str, help="Fecha/hora fin")
    parser.add_argument("--max-hops", type=int, default=15)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--max-rtt", type=float, default=500.0)
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    return parser.parse_args()


def validate_time_arguments(args):
    has_days = args.days is not None
    has_start = args.start_time is not None
    has_stop = args.stop_time is not None
    
    if has_days and (has_start or has_stop):
        print("❌ Error: No se puede combinar --days con --start-time o --stop-time")
        sys.exit(1)
    if has_start and not has_stop:
        print("❌ Error: --start-time requiere --stop-time")
        sys.exit(1)
    if has_stop and not has_start:
        print("❌ Error: --stop-time requiere --start-time")
        sys.exit(1)
    if not has_days and not has_start:
        print("❌ Error: Debe especificar --days O (--start-time y --stop-time)")
        sys.exit(1)
    
    return "days" if has_days else "range"


def parse_datetime(dt_string):
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%SZ", 
               "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(dt_string.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Formato de fecha no reconocido: '{dt_string}'")


def get_probe_metadata(probe_id):
    print(f"\n🔍 Obteniendo metadata de Probe ID {probe_id}...")
    try:
        probe = Probe(id=probe_id)
        metadata = {
            'asn_v4': probe.asn_v4 if hasattr(probe, 'asn_v4') else 'Unknown',
            'country_code': probe.country_code if hasattr(probe, 'country_code') else 'Unknown',
        }
        print(f"   ✅ ASN de la sonda: {metadata['asn_v4']}, País: {metadata['country_code']}")
        return metadata
    except Exception as e:
        print(f"   ⚠️ Error obteniendo metadata: {e}")
        return {'asn_v4': 'Unknown', 'country_code': 'Unknown'}


def descargar_historial(msm_id, probe_id, start_time, stop_time):
    print(f"\n📥 Solicitando historial de traceroute para Probe ID {probe_id}...")
    print(f"   Rango: {start_time.strftime('%Y-%m-%d %H:%M')} a {stop_time.strftime('%Y-%m-%d %H:%M')}")
    
    kwargs = {"msm_id": msm_id, "start": start_time, "stop": stop_time, "probe_ids": [probe_id]}
    is_success, results = AtlasResultsRequest(**kwargs).create()
    
    if not is_success:
        print(f"❌ Error en la solicitud: {results}")
        return None
    
    print(f"   ✅ {len(results)} ciclos de traceroute descargados.")
    return results


def procesar_traceroute(results, probe_id, msm_id, probe_metadata, max_hops, verbose=False):
    print("\n🔍 Procesando resultados de traceroute (estructura hop-by-hop)...")
    
    all_data = []
    target_addr = None
    total_hops_processed = 0
    
    for res in results:
        try:
            timestamp = res.get('timestamp')
            if timestamp is None:
                continue
                
            current_target = res.get('dst_addr') or res.get('dst_name')
            if target_addr is None:
                target_addr = current_target
            
            hops = res.get('result', [])
            if not hops:
                continue
            
            for hop_data in hops:
                hop_num = hop_data.get('hop')
                if hop_num is not None and hop_num > max_hops:
                    break
                
                hop_packets = hop_data.get('result', [])
                rtts = []
                hop_ip = "unknown"
                
                for packet in hop_packets:
                    if isinstance(packet, dict):
                        if packet.get('rtt') is not None:
                            rtts.append(float(packet['rtt']))
                        if hop_ip == "unknown" and packet.get('from'):
                            hop_ip = packet.get('from')
                
                if rtts:
                    avg_rtt = np.mean(rtts)
                    std_rtt = np.std(rtts) if len(rtts) > 1 else 0.0
                    
                    all_data.append({
                        'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        'cycle_number': None,
                        'probe_id': probe_id,
                        'probe_asn': probe_metadata['asn_v4'],
                        'country_code': probe_metadata['country_code'],
                        'target_addr': target_addr,
                        'hop_number': hop_num if hop_num is not None else 0,
                        'hop_ip': hop_ip,
                        'rtt_avg_ms': avg_rtt,
                        'rtt_min_ms': min(rtts),
                        'rtt_max_ms': max(rtts),
                        'rtt_std_ms': std_rtt,
                        'packets_received': len(rtts)
                    })
                    total_hops_processed += 1
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error procesando ciclo: {e}")
            continue
    
    if not all_data:
        print("   ❌ No se extrajeron datos válidos.")
        return None, 0
    
    df = pd.DataFrame(all_data)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df['cycle_number'] = df.index.map(lambda x: (df.index == x).argmax() + 1)
    
    cols = ['cycle_number', 'probe_id', 'probe_asn', 'country_code', 'target_addr', 
            'hop_number', 'hop_ip', 'rtt_avg_ms', 'rtt_min_ms', 'rtt_max_ms', 
            'rtt_std_ms', 'packets_received']
    df = df[cols]
    
    unique_cycles = df['cycle_number'].nunique()
    print(f"   ✅ {unique_cycles} ciclos procesados exitosamente, {total_hops_processed} saltos totales.")
    return df, total_hops_processed


def calcular_umbrales_iqr(df, k=1.5):
    umbrales = {}
    for hop_num, group in df.groupby('hop_number'):
        rtts = group['rtt_avg_ms']
        q1, q3 = rtts.quantile(0.25), rtts.quantile(0.75)
        iqr = q3 - q1
        mediana = rtts.median()
        umbral_iqr = q3 + k * iqr
        umbral_fallback = mediana * 3
        umbrales[hop_num] = max(umbral_iqr, umbral_fallback)
    return umbrales


def aplicar_deteccion_anomalias(df, umbrales):
    df = df.copy()
    df['anomaly_threshold'] = df['hop_number'].map(umbrales)
    df['is_anomaly'] = (df['rtt_avg_ms'] > df['anomaly_threshold']).astype(int)
    df['anomaly_hop'] = df.apply(
        lambda row: row['hop_number'] if row['is_anomaly'] == 1 else None, axis=1
    )
    df['anomaly_ratio'] = df['rtt_avg_ms'] / df['anomaly_threshold']
    return df


def validar_y_guardar(df, probe_id, msm_id, max_rtt, output_file, stats_por_hop=None):
    print("\n⚙️  Validando y calculando métricas de serie temporal...")
    
    df_filtered = df[df['rtt_avg_ms'] < max_rtt].copy()
    if output_file is None:
        output_file = f"historial_traceroute_probe_{probe_id}.csv"
    
    df_filtered.to_csv(output_file, index=True)
    print(f"💾 Datos guardados en: {output_file}")
    
    # Resumen de umbrales
    if stats_por_hop:
        print("\n📊 Umbrales IQR adaptativos calculados:")
        print(f"   {'Hop':<5} {'Umbral':<10} {'Muestras':<10}")
        for hop_num in sorted(stats_por_hop.keys()):
            count = len(df[df['hop_number'] == hop_num])
            print(f"   {hop_num:<5} {stats_por_hop[hop_num]:<10.2f} {count:<10}")
    
    # Resumen de anomalías
    anomalias = df_filtered[df_filtered['is_anomaly'] == 1]
    total_anomalias = len(anomalias)
    ciclos_anomalos = anomalias['cycle_number'].nunique() if total_anomalias > 0 else 0
    total_ciclos = df_filtered['cycle_number'].nunique()
    
    print(f"\n🚨 Resumen de Anomalías Detectadas:")
    if total_ciclos > 0:
        print(f"   Ciclos con anomalías: {ciclos_anomalos} de {total_ciclos} "
              f"({100*ciclos_anomalos/total_ciclos:.2f}%)")
    else:
        print("   Sin datos para analizar")
    
    # Generación de URLs de Path Analysis (CORREGIDA)
    if ciclos_anomalos > 0:
        print(f"\n🔗 URLs de RIPE Atlas Path Analysis para validación visual:")
        
        # CORRECCIÓN: reset_index() convierte el índice 'timestamp' en columna
        anomalias_reset = anomalias.reset_index()
        
        ciclos_anom_info = (
            anomalias_reset.groupby('cycle_number')
            .agg({
                'timestamp': 'first',
                'anomaly_hop': lambda x: int(x.mode()[0]) if len(x) > 0 else None,
                'rtt_avg_ms': 'max',
                'anomaly_threshold': 'min'
            })
            .sort_values('timestamp')
        )
        
        max_urls = min(10, len(ciclos_anom_info))
        for cycle_num, info in ciclos_anom_info.head(max_urls).iterrows():
            ts = info['timestamp']
            if hasattr(ts, 'timestamp'):
                ts_ms = int(ts.timestamp() * 1000)
            else:
                ts_ms = int(pd.to_datetime(ts).timestamp() * 1000)
            
            window_ms = 7200000
            url = (f"https://atlas.ripe.net/pathanalysis/embed?"
                   f"measurementId={msm_id}&"
                   f"sourceProbeId={probe_id}&"
                   f"center={ts_ms}&"
                   f"window={window_ms}")
            
            print(f"\n   Ciclo {int(cycle_num)} ({ts.strftime('%Y-%m-%d %H:%M UTC')}):")
            print(f"      Hop anómalo: {info['anomaly_hop']} | "
                  f"RTT máx: {info['rtt_avg_ms']:.2f}ms | "
                  f"Umbral: {info['anomaly_threshold']:.2f}ms")
            print(f"      URL: {url}")
        
        if len(ciclos_anom_info) > max_urls:
            print(f"\n   ... y {len(ciclos_anom_info) - max_urls} ciclos anómalos más "
                  f"(revisa el CSV para ver todos)")

    # Validación de Serie Temporal
    print("\n🔬 Validando propiedades de Serie Temporal (Ground Truth):")
    max_hop = df_filtered['hop_number'].max()
    e2e_df = df_filtered[df_filtered['hop_number'] == max_hop].copy()
    
    print(f"   Probe: {probe_id} ({df_filtered['country_code'].iloc[0]}, "
          f"ASN {df_filtered['probe_asn'].iloc[0]})")
    print(f"   Target: {df_filtered['target_addr'].iloc[0]} | Ciclos: {total_ciclos:,}")
    
    if len(e2e_df) > 10:
        print(f"\n   📊 Métricas End-to-End (Salto {max_hop}):")
        print(f"      Mediana RTT: {e2e_df['rtt_avg_ms'].median():.2f} ms")
        print(f"      Percentil 95 RTT: {e2e_df['rtt_avg_ms'].quantile(0.95):.2f} ms")
        
        # CORRECCIÓN: Cálculo robusto de autocorrelación Spearman
        rtt_series = e2e_df['rtt_avg_ms']
        rtt_shifted = rtt_series.shift(1)
        valid_pairs = pd.concat([rtt_series, rtt_shifted], axis=1).dropna()
        
        if len(valid_pairs) > 10:
            autocorr_spearman = spearmanr(valid_pairs.iloc[:, 0], valid_pairs.iloc[:, 1])[0]
            autocorr_pearson = rtt_series.autocorr(lag=1)
            
            print(f"\n   📊 Autocorrelación Temporal (End-to-End):")
            print(f"      Spearman (Lag-1): {autocorr_spearman:.4f} "
                  f"{'✅' if autocorr_spearman > 0.3 else '⚠️'}")
            print(f"      Pearson (Lag-1): {autocorr_pearson:.4f} "
                  f"{'✅' if autocorr_pearson > 0.3 else '⚠️'}")
            
            diff = abs(autocorr_spearman - autocorr_pearson)
            if diff > 0.2:
                print(f"      ℹ️  Diferencia significativa: Spearman es más confiable por posibles outliers.")
        
        print(f"\n   📊 Estadísticas de Jitter (Intra-salto, std dev):")
        print(f"      Mediana: {df_filtered['rtt_std_ms'].median():.3f} ms")
        print(f"      Percentil 95: {df_filtered['rtt_std_ms'].quantile(0.95):.3f} ms")


def main():
    args = parse_arguments()
    time_mode = validate_time_arguments(args)
    
    print("="*70)
    print("EXTRACCIÓN DE HISTORIAL DE TRACEROUTE (RIPE Atlas)")
    print("con Detección Adaptativa de Anomalías por Salto (IQR Tukey)")
    print("="*70)
    print(f"\n⚙️  Configuración:")
    print(f"   Measurement ID: {args.measurement_id} | Probe ID: {args.probe_id}")
    print(f"   Máx. saltos: {args.max_hops} | Filtro max RTT: {args.max_rtt} ms | IQR k: {args.iqr_k}")
    
    if time_mode == "days":
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=args.days)
        print(f"   Modo: Últimos {args.days} días")
    else:
        start_time = parse_datetime(args.start_time)
        end_time = parse_datetime(args.stop_time)
        print(f"   Modo: Rango de fechas específico")
    
    print(f"   Período: {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    probe_metadata = get_probe_metadata(args.probe_id)
    resultados = descargar_historial(args.measurement_id, args.probe_id, start_time, end_time)
    
    if not resultados:
        print("\n❌ No se pudieron descargar los resultados.")
        sys.exit(1)
    
    df, _ = procesar_traceroute(
        resultados, args.probe_id, args.measurement_id, 
        probe_metadata, args.max_hops, args.verbose
    )
    if df is None:
        sys.exit(1)
    
    print(f"\n📐 Calculando umbrales IQR adaptativos por salto (k={args.iqr_k})...")
    umbrales = calcular_umbrales_iqr(df, k=args.iqr_k)
    
    print(f"   🔎 Aplicando detección de anomalías...")
    df = aplicar_deteccion_anomalias(df, umbrales)
    total_anomalias = df['is_anomaly'].sum()
    print(f"   ✅ {total_anomalias} filas marcadas como anómalas ({100*total_anomalias/len(df):.2f}%)")
    
    validar_y_guardar(df, args.probe_id, args.measurement_id, args.max_rtt, args.output, umbrales)
    print("\n✅ ¡Proceso completado!")


if __name__ == "__main__":
    main()
