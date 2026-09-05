#!/usr/bin/env python3
"""
Script para extraer datos de latencia y jitter de RIPE Atlas
para validación empírica del modelo generativo de latencia BGP.

Soporta mediciones tipo Ping y Traceroute.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from ripe.atlas.cousteau import AtlasResultsRequest


# ==========================================
# CONFIGURACIÓN POR DEFECTO
# ==========================================
DEFAULT_TARGET = "185.136.96.1"  # RIPE Atlas anchor (buen coverage)
DEFAULT_DAYS_BACK = 30
DEFAULT_MEASUREMENT_TYPE = "ping"  # "ping" o "traceroute"


# ==========================================
# PARSING DE ARGUMENTOS
# ==========================================
def parse_arguments():
    """
    Define y parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Extracción de métricas de latencia y jitter desde RIPE Atlas "
                    "para validación empírica de modelos generativos BGP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --measurement-id 208230991
  %(prog)s --measurement-id 208230991 --type traceroute --days 60
  %(prog)s --measurement-id 208230991 --output my_data.csv
  %(prog)s --list-types  # Muestra los tipos de medición soportados
        """,
    )

    parser.add_argument(
        "--measurement-id",
        "-m",
        type=int,
        required=True,
        help="ID de la medición pública de RIPE Atlas a consultar.",
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["ping", "traceroute"],
        default=DEFAULT_MEASUREMENT_TYPE,
        help="Tipo de medición a procesar (default: %(default)s).",
    )

    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help="Número de días hacia atrás para extraer datos (default: %(default)s).",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Target esperado de la medición (solo informativo, "
            "no se usa para filtrar). Default: autodetectado."
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Archivo CSV de salida. "
            "Default: ripe_atlas_<type>_<measurement_id>_<days>d.csv"
        ),
    )

    parser.add_argument(
        "--max-rtt",
        type=float,
        default=500.0,
        help="Filtro de outlier: descartar RTT mayores a este valor en ms (default: %(default)s).",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mostrar información detallada durante la ejecución.",
    )

    return parser.parse_args()


# ==========================================
# EXTRACCIÓN DE DATOS
# ==========================================
def download_results(measurement_id, days_back, verbose=False):
    """
    Descarga los resultados de una medición existente de RIPE Atlas.
    """
    print(f"\n📥 Descargando resultados de la medición {measurement_id}...")

    end_time = int(datetime.now(timezone.utc).timestamp())
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())

    print(
        f"   Consultando desde "
        f"{datetime.fromtimestamp(start_time, tz=timezone.utc).strftime('%Y-%m-%d')} "
        f"hasta ahora..."
    )

    is_success, results = AtlasResultsRequest(
        msm_id=measurement_id,
        start=start_time,
        stop=end_time,
    ).create()

    if not is_success:
        print(f" Error al descargar resultados: {results}")
        return None, None

    print(f"   Procesando resultados...")
    return results, measurement_id


# ==========================================
# PARSER PARA PING
# ==========================================
def parse_ping_results(results, verbose=False):
    """
    Parsea resultados de medición tipo Ping.
    Retorna: (lista_de_muestras, metadata)
    """
    all_results = []
    target = None

    for result_data in results:
        try:
            probe_id = result_data.get("prb_id")
            asn = result_data.get("asn", "Unknown")
            timestamp = result_data.get("timestamp")
            if target is None:
                target = result_data.get("dst_name") or result_data.get("dst_addr")

            # Cada medición ping contiene una lista de paquetes en "result"
            for packet in result_data.get("result", []):
                if "rtt" in packet and packet["rtt"] is not None:
                    all_results.append({
                        "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        "rtt_ms": float(packet["rtt"]),
                        "probe_id": probe_id,
                        "asn": asn,
                        "hop": None,  # No aplica en ping
                        "jitter_ms": None,  # Se calcula post-procesamiento
                    })
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error procesando resultado ping: {e}")
            continue

    metadata = {"type": "ping", "target": target, "total_probes": len(set(r["probe_id"] for r in all_results))}
    return all_results, metadata


# ==========================================
# PARSER PARA TRACEROUTE
# ==========================================
def parse_traceroute_results(results, max_hops=15, verbose=False):
    """
    Parsea resultados de medición tipo Traceroute.
    
    Para cada medición, extrae:
      - RTT end-to-end (suma o último hop con RTT válido)
      - Jitter inter-hop (variación entre RTTs de hops consecutivos)
      - Jitter intra-hop (variación entre los 3 RTTs del mismo hop)
    """
    all_results = []
    target = None

    for result_data in results:
        try:
            probe_id = result_data.get("prb_id")
            asn = result_data.get("asn", "Unknown")
            timestamp = result_data.get("timestamp")
            if target is None:
                target = result_data.get("dst_name") or result_data.get("dst_addr")

            hops = result_data.get("result", [])
            if not hops:
                continue

            # Extraer RTTs por hop
            hop_rtts = []  # lista de (hop_index, rtt_promedio_del_hop)
            for hop in hops[:max_hops]:
                hop_index = hop.get("hop", len(hop_rtts) + 1)
                hop_results = hop.get("result", [])
                
                # Filtrar RTTs válidos del hop (puede haber 3 intentos)
                valid_rtts = [
                    float(r["rtt"]) 
                    for r in hop_results 
                    if isinstance(r, dict) and "rtt" in r and r["rtt"] is not None
                ]
                
                if valid_rtts:
                    avg_rtt = np.mean(valid_rtts)
                    hop_rtts.append((hop_index, avg_rtt, valid_rtts))

            if not hop_rtts:
                continue

            # RTT end-to-end: RTT del último hop con medición válida
            last_hop_index, last_hop_rtt, last_hop_rtts = hop_rtts[-1]
            all_results.append({
                "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                "rtt_ms": float(last_hop_rtt),
                "probe_id": probe_id,
                "asn": asn,
                "hop": last_hop_index,
                "jitter_ms": None,  # Se calcula abajo
            })

            # Calcular jitter inter-hop (variación entre hops consecutivos)
            for i in range(1, len(hop_rtts)):
                prev_idx, prev_rtt, _ = hop_rtts[i - 1]
                curr_idx, curr_rtt, _ = hop_rtts[i]
                inter_hop_jitter = abs(curr_rtt - prev_rtt)
                all_results.append({
                    "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                    "rtt_ms": float(curr_rtt),
                    "probe_id": probe_id,
                    "asn": asn,
                    "hop": curr_idx,
                    "jitter_ms": float(inter_hop_jitter),
                })

            # Calcular jitter intra-hop (variación entre los 3 intentos del mismo hop)
            for hop_index, _, rtts in hop_rtts:
                if len(rtts) >= 2:
                    intra_jitter = float(np.std(rtts))
                    all_results.append({
                        "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        "rtt_ms": float(np.mean(rtts)),
                        "probe_id": probe_id,
                        "asn": asn,
                        "hop": hop_index,
                        "jitter_ms": float(intra_jitter),
                        "jitter_type": "intra-hop",
                    })

        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error procesando resultado traceroute: {e}")
            continue

    metadata = {
        "type": "traceroute",
        "target": target,
        "total_probes": len(set(r["probe_id"] for r in all_results)),
        "max_hops_analyzed": max_hops,
    }
    return all_results, metadata


# ==========================================
# POST-PROCESAMIENTO: CÁLCULO DE JITTER TEMPORAL
# ==========================================
def compute_temporal_jitter(df):
    """
    Calcula el jitter temporal (variación entre mediciones sucesivas)
    para cada sonda. Esto es el jitter clásico definido por RFC 3550.
    """
    df = df.copy()
    df = df.sort_values(["probe_id", "timestamp"])
    
    # Jitter temporal: diferencia absoluta entre RTTs consecutivos de la misma sonda
    df["jitter_temporal_ms"] = (
        df.groupby("probe_id")["rtt_ms"]
        .diff()
        .abs()
    )
    
    # Si no hay jitter_ms calculado (ej. en ping), usar el jitter temporal
    if "jitter_ms" not in df.columns or df["jitter_ms"].isna().all():
        df["jitter_ms"] = df["jitter_temporal_ms"]
    
    return df


# ==========================================
# ANÁLISIS Y GUARDADO
# ==========================================
def analyze_and_save(results, metadata, max_rtt, output_file=None):
    """
    Analiza los resultados, calcula jitter temporal y guarda en CSV.
    """
    if not results:
        print("❌ No hay resultados para guardar")
        return None

    df = pd.DataFrame(results)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Filtrar outliers
    df = df[df["rtt_ms"] < max_rtt].copy()

    # Calcular jitter temporal
    df = compute_temporal_jitter(df)

    # Generar nombre de archivo por defecto si no se proporcionó
    if output_file is None:
        mtype = metadata.get("type", "unknown")
        mid = metadata.get("measurement_id", "unknown")
        output_file = f"ripe_atlas_{mtype}_{mid}.csv"

    print(f"\n📊 Estadísticas básicas de latencia RTT:")
    print(f"   Tipo de medición: {metadata.get('type')}")
    print(f"   Target: {metadata.get('target')}")
    print(f"   Total de muestras: {len(df)}")
    print(f"   Sondas únicas: {metadata.get('total_probes', 'N/A')}")
    print(f"   Mediana RTT: {df['rtt_ms'].median():.2f} ms")
    print(f"   Percentil 95 RTT: {df['rtt_ms'].quantile(0.95):.2f} ms")
    print(f"   Máximo RTT: {df['rtt_ms'].max():.2f} ms")
    print(f"   Mínimo RTT: {df['rtt_ms'].min():.2f} ms")
    print(f"   Desviación estándar RTT: {df['rtt_ms'].std():.2f} ms")

    # Estadísticas de jitter (si existen valores no nulos)
    jitter_col = "jitter_ms"
    if jitter_col in df.columns and df[jitter_col].notna().any():
        jitter_valid = df[jitter_col].dropna()
        print(f"\n📊 Estadísticas de Jitter:")
        print(f"   Mediana Jitter: {jitter_valid.median():.2f} ms")
        print(f"   Percentil 95 Jitter: {jitter_valid.quantile(0.95):.2f} ms")
        print(f"   Máximo Jitter: {jitter_valid.max():.2f} ms")

    df.to_csv(output_file, index=True)
    print(f"\n💾 Datos guardados en '{output_file}'")

    return df


# ==========================================
# MÉTRICAS EMPÍRICAS
# ==========================================
def calculate_empirical_metrics(df):
    """
    Calcula las métricas empíricas para validar el modelo generativo.
    """
    print("\n🔬 Calculando métricas empíricas para validación...")

    rtt = df["rtt_ms"].dropna()

    metrics = {
        "median": float(np.median(rtt)),
        "p95": float(np.percentile(rtt, 95)),
        "p99": float(np.percentile(rtt, 99)),
        "max": float(np.max(rtt)),
        "min": float(np.min(rtt)),
        "std": float(np.std(rtt)),
        "mean": float(np.mean(rtt)),
    }

    # Variación diurna
    df["hour"] = df.index.hour
    df["is_peak"] = df["hour"].apply(lambda x: 1 if (18 <= x <= 23) or (8 <= x <= 10) else 0)

    median_offpeak = df[df["is_peak"] == 0]["rtt_ms"].median()
    median_peak = df[df["is_peak"] == 1]["rtt_ms"].median()
    
    if pd.notna(median_offpeak) and median_offpeak > 0:
        metrics["diurnal_var_pct"] = float(((median_peak - median_offpeak) / median_offpeak) * 100)
        metrics["median_offpeak_ms"] = float(median_offpeak)
        metrics["median_peak_ms"] = float(median_peak)
    else:
        metrics["diurnal_var_pct"] = float("nan")
        metrics["median_offpeak_ms"] = float("nan")
        metrics["median_peak_ms"] = float("nan")

    # Autocorrelación (Lag-1)
    df_resampled = (
        df.groupby([pd.Grouper(freq="1min"), "probe_id"])["rtt_ms"]
        .mean()
        .reset_index()
    )
    if len(df_resampled) > 1:
        try:
            df_pivot = df_resampled.pivot(
                index="timestamp", columns="probe_id", values="rtt_ms"
            ).mean(axis=1)
            metrics["autocorr_lag1"] = float(df_pivot.autocorr(lag=1))
        except Exception:
            metrics["autocorr_lag1"] = float("nan")
    else:
        metrics["autocorr_lag1"] = float("nan")

    # Estadísticas de jitter
    if "jitter_ms" in df.columns and df["jitter_ms"].notna().any():
        jitter = df["jitter_ms"].dropna()
        metrics["jitter_median"] = float(jitter.median())
        metrics["jitter_p95"] = float(jitter.quantile(0.95))
        metrics["jitter_mean"] = float(jitter.mean())

    print(f"\n--- MÉTRICAS EMPÍRICAS ---")
    print(f"Mediana RTT: {metrics['median']:.2f} ms")
    print(f"Percentil 95 RTT: {metrics['p95']:.2f} ms")
    print(f"Percentil 99 RTT: {metrics['p99']:.2f} ms")
    print(f"Variación Diurna: {metrics['diurnal_var_pct']:.1f}%" if not np.isnan(metrics['diurnal_var_pct']) else "Variación Diurna: N/A")
    print(f"Autocorrelación (Lag-1): {metrics['autocorr_lag1']:.4f}" if not np.isnan(metrics['autocorr_lag1']) else "Autocorrelación (Lag-1): N/A")
    
    if "jitter_median" in metrics:
        print(f"Mediana Jitter: {metrics['jitter_median']:.2f} ms")
        print(f"Percentil 95 Jitter: {metrics['jitter_p95']:.2f} ms")

    return metrics


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    args = parse_arguments()

    print("=" * 70)
    print("RIPE Atlas - Extracción de Datos de Latencia y Jitter")
    print("para Validación Empírica del Modelo Generativo BGP")
    print("=" * 70)
    print(f"\n⚙️  Configuración:")
    print(f"   Measurement ID: {args.measurement_id}")
    print(f"   Tipo: {args.type}")
    print(f"   Días hacia atrás: {args.days}")
    print(f"   Filtro max RTT: {args.max_rtt} ms")

    # 1. Descargar resultados
    results, _ = download_results(
        args.measurement_id, args.days, verbose=args.verbose
    )

    if not results:
        print("⚠️ No se pudieron descargar resultados.")
        sys.exit(1)

    # 2. Parsear según tipo
    print(f"\n🔍 Parseando resultados tipo '{args.type}'...")
    if args.type == "ping":
        parsed_results, metadata = parse_ping_results(results, verbose=args.verbose)
    elif args.type == "traceroute":
        parsed_results, metadata = parse_traceroute_results(results, verbose=args.verbose)
    else:
        print(f"❌ Tipo de medición no soportado: {args.type}")
        sys.exit(1)

    metadata["measurement_id"] = args.measurement_id

    if not parsed_results:
        print("️ No se pudieron parsear resultados válidos.")
        sys.exit(1)

    print(f"   ✅ {len(parsed_results)} muestras extraídas")

    # 3. Analizar y guardar
    df = analyze_and_save(
        parsed_results,
        metadata,
        max_rtt=args.max_rtt,
        output_file=args.output,
    )

    if df is not None:
        # 4. Calcular métricas empíricas
        empirical_metrics = calculate_empirical_metrics(df)

        print("\n✅ Proceso completado exitosamente!")
        print("💡 Usa estas métricas para validar tu modelo generativo:")
        print(f"   - Mediana objetivo: {empirical_metrics['median']:.2f} ms")
        print(f"   - Percentil 95 objetivo: {empirical_metrics['p95']:.2f} ms")
        if not np.isnan(empirical_metrics.get("diurnal_var_pct", float("nan"))):
            print(f"   - Variación diurna objetivo: {empirical_metrics['diurnal_var_pct']:.1f}%")
        if "jitter_median" in empirical_metrics:
            print(f"   - Jitter mediana objetivo: {empirical_metrics['jitter_median']:.2f} ms")


if __name__ == "__main__":
    main()
