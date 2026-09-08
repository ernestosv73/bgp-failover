#!/usr/bin/env python3
"""
Script para extraer datos de latencia y jitter de RIPE Atlas
para validación empírica del modelo generativo de latencia BGP.
Soporta mediciones tipo Ping y Traceroute, IPv4 e IPv6.
Incluye resolución de ASN y country_code desde metadata de sondas.
"""
import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from ripe.atlas.cousteau import AtlasResultsRequest, Probe, MeasurementRequest

# ==========================================
# CONFIGURACIÓN POR DEFECTO
# ==========================================
DEFAULT_DAYS_BACK = 30
DEFAULT_MEASUREMENT_TYPE = "ping"

# Cache global para metadata de sondas
probe_metadata_cache = {}

# ==========================================
# PARSING DE ARGUMENTOS
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extracción de métricas de latencia y jitter desde RIPE Atlas (IPv4/IPv6)"
    )
    parser.add_argument("--measurement-id", "-m", type=int, required=True)
    parser.add_argument("--type", "-t", choices=["ping", "traceroute"], default="ping")
    parser.add_argument("--days", "-d", type=int, default=3)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--max-rtt", type=float, default=500.0)
    parser.add_argument("--af", type=int, choices=[4, 6], default=None,
                        help="Address family: 4=IPv4, 6=IPv6. Auto-detectado si no se especifica.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()

# ==========================================
# DETECCIÓN AUTOMÁTICA DE AF
# ==========================================
def detect_measurement_af(measurement_id):
    """Detecta el address family (4 o 6) de una medición consultando su metadata."""
    try:
        for msm in MeasurementRequest(id=measurement_id):
            af = msm.get('af', 4)
            return af
    except Exception as e:
        print(f"   ⚠️ No se pudo detectar AF automáticamente: {e}")
    return 4

# ==========================================
# RESOLUCIÓN DE METADATA DE SONDA (CORREGIDA)
# ==========================================
def get_probe_metadata(probe_id, af=4):
    """
    Obtiene metadata de una sonda (ASN, country_code) usando la API de RIPE Atlas.
    CORRECCIÓN: Implementa fallback inteligente entre ASN v4/v6.
    """
    global probe_metadata_cache
    
    cache_key = f"{probe_id}_v{af}"
    if cache_key in probe_metadata_cache:
        return probe_metadata_cache[cache_key]
    
    try:
        probe = Probe(id=probe_id)
        
        # Obtener ambos ASN de forma segura
        asn_v4 = getattr(probe, 'asn_v4', None)
        asn_v6 = getattr(probe, 'asn_v6', None)
        country = getattr(probe, 'country_code', 'Unknown')
        is_anchor = getattr(probe, 'is_anchor', False)
        address_v4 = getattr(probe, 'address_v4', None)
        address_v6 = getattr(probe, 'address_v6', None)
        
        # CORRECCIÓN CLAVE: Fallback inteligente
        # Si el AF solicitado no tiene ASN, usar el otro
        if af == 6:
            asn = asn_v6 if asn_v6 else (asn_v4 if asn_v4 else 'Unknown')
            address = address_v6 if address_v6 else address_v4
        else:
            asn = asn_v4 if asn_v4 else (asn_v6 if asn_v6 else 'Unknown')
            address = address_v4 if address_v4 else address_v6
        
        metadata = {
            'asn': asn,
            'asn_v4': asn_v4 if asn_v4 else 'Unknown',
            'asn_v6': asn_v6 if asn_v6 else 'Unknown',
            'country_code': country if country else 'Unknown',
            'is_anchor': is_anchor,
            'address': address,
            'address_v4': address_v4,
            'address_v6': address_v6,
        }
        
        probe_metadata_cache[cache_key] = metadata
        return metadata
        
    except Exception as e:
        print(f"   ⚠️ Error obteniendo metadata de probe {probe_id}: {e}")
        return {
            'asn': 'Unknown',
            'asn_v4': 'Unknown',
            'asn_v6': 'Unknown',
            'country_code': 'Unknown',
            'is_anchor': False,
            'address': None,
            'address_v4': None,
            'address_v6': None,
        }

# ==========================================
# EXTRACCIÓN DE DATOS
# ==========================================
def download_results(measurement_id, days_back, verbose=False):
    print(f"\n📥 Descargando resultados de la medición {measurement_id}...")
    
    end_time = int(datetime.now(timezone.utc).timestamp())
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    
    print(f"   Consultando desde {datetime.fromtimestamp(start_time, tz=timezone.utc).strftime('%Y-%m-%d')} hasta ahora...")
    
    is_success, results = AtlasResultsRequest(
        msm_id=measurement_id,
        start=start_time,
        stop=end_time,
    ).create()
    
    if not is_success:
        print(f"   ❌ Error: {results}")
        return None
    
    print(f"   ✅ {len(results)} mediciones descargadas")
    return results

# ==========================================
# PARSER PARA PING (CORREGIDO)
# ==========================================
def parse_ping_results(results, af=4, verbose=False):
    print(f"\n🔍 Procesando resultados (IPv{af}) y resolviendo metadata de sondas...")
    
    all_results = []
    target = None
    unique_probes = set()
    
    # Extraer probe_ids únicos
    for result_data in results:
        probe_id = result_data.get("prb_id")
        if probe_id:
            unique_probes.add(probe_id)
    
    print(f"   📊 Sondas únicas encontradas: {len(unique_probes)}")
    print(f"   🔄 Resolviendo metadata para {len(unique_probes)} sondas...")
    
    # Precargar metadata con verbose detallado
    probe_metadata = {}
    for i, probe_id in enumerate(unique_probes, 1):
        metadata = get_probe_metadata(probe_id, af=af)
        probe_metadata[probe_id] = metadata
        
        if verbose:
            print(f"      [{i}/{len(unique_probes)}] Probe {probe_id}: "
                  f"ASN={metadata['asn']}, CC={metadata['country_code']}, "
                  f"ASNv4={metadata['asn_v4']}, ASNv6={metadata['asn_v6']}")
    
    # Procesar resultados
    processed_count = 0
    skipped_no_rtt = 0
    skipped_error = 0
    
    for result_data in results:
        try:
            probe_id = result_data.get("prb_id")
            timestamp = result_data.get("timestamp")
            
            if timestamp is None:
                skipped_error += 1
                continue
            
            if target is None:
                target = result_data.get("dst_name") or result_data.get("dst_addr")
            
            metadata = probe_metadata.get(probe_id, {})
            asn = metadata.get('asn', 'Unknown')
            country = metadata.get('country_code', 'Unknown')
            
            # Extraer RTTs del ciclo
            rtts = []
            for packet in result_data.get("result", []):
                if "rtt" in packet and packet["rtt"] is not None:
                    rtts.append(float(packet["rtt"]))
            
            if len(rtts) >= 1:
                avg_rtt = np.mean(rtts)
                std_rtt = np.std(rtts) if len(rtts) > 1 else 0.0
                
                all_results.append({
                    'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                    'cycle_number': None,
                    'probe_id': probe_id,
                    'asn': asn,
                    'country_code': country,
                    'target': target,
                    'rtt_avg_ms': avg_rtt,
                    'rtt_std_ms': std_rtt,
                    'rtt_min_ms': min(rtts),
                    'rtt_max_ms': max(rtts),
                    'packets_count': len(rtts)
                })
                processed_count += 1
            else:
                skipped_no_rtt += 1
                
        except Exception as e:
            skipped_error += 1
            if verbose:
                print(f"   ⚠️ Error procesando resultado: {e}")
            continue
    
    print(f"   📊 Procesados: {processed_count} | Sin RTT: {skipped_no_rtt} | Errores: {skipped_error}")
    
    if not all_results:
        print("   ❌ No se extrajeron datos válidos")
        if verbose:
            print("   💡 Tip: Ejecuta con --verbose para ver detalles del procesamiento")
        return None
    
    # Crear DataFrame
    df = pd.DataFrame(all_results)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df['cycle_number'] = range(1, len(df) + 1)
    
    df = df[['cycle_number', 'probe_id', 'asn', 'country_code', 'target',
             'rtt_avg_ms', 'rtt_std_ms', 'rtt_min_ms', 'rtt_max_ms', 'packets_count']]
    
    print(f"   ✅ {len(df)} ciclos procesados")
    
    # Resumen por sonda
    print(f"\n    Resumen por sonda (IPv{af}):")
    probe_summary = df.groupby(['probe_id', 'asn', 'country_code']).agg({
        'cycle_number': 'count',
        'rtt_avg_ms': 'median'
    }).reset_index()
    probe_summary.columns = ['probe_id', 'asn', 'country_code', 'ciclos', 'mediana_rtt']
    
    for _, row in probe_summary.iterrows():
        print(f"      Probe {row['probe_id']} ({row['country_code']}, ASN {row['asn']}): "
              f"{int(row['ciclos'])} ciclos, mediana {row['mediana_rtt']:.2f} ms")
    
    return df

# ==========================================
# PARSER PARA TRACEROUTE
# ==========================================
def parse_traceroute_results(results, af=4, max_hops=15, verbose=False):
    print(f"\n🔍 Procesando resultados de traceroute (IPv{af})...")
    
    all_results = []
    target = None
    unique_probes = set()
    
    for result_data in results:
        probe_id = result_data.get("prb_id")
        if probe_id:
            unique_probes.add(probe_id)
    
    print(f"   📊 Sondas únicas: {len(unique_probes)}")
    print(f"   🔄 Resolviendo metadata...")
    
    probe_metadata = {}
    for probe_id in unique_probes:
        probe_metadata[probe_id] = get_probe_metadata(probe_id, af=af)
    
    for result_data in results:
        try:
            probe_id = result_data.get("prb_id")
            timestamp = result_data.get("timestamp")
            
            if target is None:
                target = result_data.get("dst_name") or result_data.get("dst_addr")
            
            metadata = probe_metadata.get(probe_id, {})
            asn = metadata.get('asn', 'Unknown')
            country = metadata.get('country_code', 'Unknown')
            
            hops = result_data.get("result", [])
            if not hops:
                continue
            
            hop_rtts = []
            for hop in hops[:max_hops]:
                hop_index = hop.get("hop", len(hop_rtts) + 1)
                hop_results = hop.get("result", [])
                
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
            
            last_hop_index, last_hop_rtt, _ = hop_rtts[-1]
            all_results.append({
                'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                'cycle_number': None,
                'probe_id': probe_id,
                'asn': asn,
                'country_code': country,
                'target': target,
                'rtt_avg_ms': last_hop_rtt,
                'rtt_std_ms': 0.0,
                'rtt_min_ms': last_hop_rtt,
                'rtt_max_ms': last_hop_rtt,
                'packets_count': len(hop_rtts[-1][2])
            })
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error: {e}")
            continue
    
    if not all_results:
        print("   ❌ No se extrajeron datos válidos")
        return None
    
    df = pd.DataFrame(all_results)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df['cycle_number'] = range(1, len(df) + 1)
    
    df = df[['cycle_number', 'probe_id', 'asn', 'country_code', 'target',
             'rtt_avg_ms', 'rtt_std_ms', 'rtt_min_ms', 'rtt_max_ms', 'packets_count']]
    
    print(f"   ✅ {len(df)} ciclos procesados")
    return df

# ==========================================
# GUARDADO (CORREGIDO PARA IPV6)
# ==========================================
def sanitize_filename(target, af=4):
    """Sanitiza el target para usarlo como nombre de archivo, soportando IPv6."""
    if not target:
        return "data"
    
    # Reemplazar caracteres problemáticos
    # IPv6: "2001:67c:2e8:22::c:40" -> "2001_67c_2e8_22__c_40"
    safe = target.replace(':', '_').replace('/', '_').replace('[', '').replace(']', '')
    
    # Limitar longitud
    if len(safe) > 50:
        safe = safe[:50]
    
    return safe

def save_to_csv(df, output_file, max_rtt, af=4):
    df_filtered = df[df['rtt_avg_ms'] < max_rtt].copy()
    
    if output_file is None:
        target = df_filtered['target'].iloc[0] if len(df_filtered) > 0 and 'target' in df_filtered.columns else 'data'
        target_safe = sanitize_filename(target, af)
        output_file = f"ripe_atlas_v{af}_{target_safe}.csv"
    
    df_filtered.to_csv(output_file, index=True)
    
    print(f"\n💾 Datos guardados en: {output_file}")
    print(f"   Total ciclos (sin outliers): {len(df_filtered)}")
    print(f"   RTT avg: {df_filtered['rtt_avg_ms'].median():.2f} ms (mediana)")
    print(f"   RTT std: {df_filtered['rtt_std_ms'].median():.2f} ms (mediana)")
    
    if 'country_code' in df_filtered.columns:
        print(f"\n📊 Distribución por país:")
        country_stats = df_filtered.groupby('country_code').agg({
            'cycle_number': 'count',
            'rtt_avg_ms': 'median'
        }).reset_index()
        country_stats.columns = ['País', 'Ciclos', 'Mediana_RTT']
        for _, row in country_stats.iterrows():
            print(f"   {row['País']}: {int(row['Ciclos'])} ciclos, {row['Mediana_RTT']:.2f} ms")
    
    return output_file

# ==========================================
# MAIN
# ==========================================
def main():
    args = parse_arguments()
    
    # Detectar AF automáticamente si no se especificó
    af = args.af
    if af is None:
        print(f"\n🔍 Detectando address family de la medición {args.measurement_id}...")
        af = detect_measurement_af(args.measurement_id)
        print(f"   ✅ Address family detectado: IPv{af}")
    
    print("="*70)
    print("RIPE Atlas - Extracción de Datos de Latencia")
    print("="*70)
    print(f"\n⚙️  Configuración:")
    print(f"   Measurement ID: {args.measurement_id}")
    print(f"   Tipo: {args.type}")
    print(f"   Address Family: IPv{af}")
    print(f"   Días: {args.days}")
    print(f"   Filtro max RTT: {args.max_rtt} ms")
    
    # Descargar
    results = download_results(args.measurement_id, args.days)
    if not results:
        sys.exit(1)
    
    # Parsear según tipo
    if args.type == "ping":
        df = parse_ping_results(results, af=af, verbose=args.verbose)
    else:
        df = parse_traceroute_results(results, af=af, verbose=args.verbose)
    
    if df is None:
        print("\n❌ No se pudieron procesar los resultados.")
        print(" Sugerencia: Ejecuta con --verbose para ver detalles del error.")
        sys.exit(1)
    
    # Guardar
    output_file = save_to_csv(df, args.output, args.max_rtt, af=af)
    
    print("\n✅ Proceso completado!")
    print(f"\n📝 Próximo paso:")
    print(f"   python3 extract_single_probe_history.py \\")
    print(f"       --measurement-id {args.measurement_id} \\")
    print(f"       --probe-id <probe_id> \\")
    print(f"       --days 30")

if __name__ == "__main__":
    main()
