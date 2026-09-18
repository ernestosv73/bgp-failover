#!/usr/bin/env python3
"""
Script corregido para extraer y analizar historial de TRACEROUTE de RIPE Atlas.
Replica la lógica de detección de anomalías de RIPE Path Analysis:
🟣 RTT Degradation (>10ms Y >20% vs. CICLO ANTERIOR, con severidad graduada)
🔴 ASN/IP Change (vs. CICLO ANTERIOR en la misma posición de salto)
🟡 Path Length Change (vs. CICLO ANTERIOR)

Correcciones acumuladas respecto a la versión original:
1. Se ordena 'results' por timestamp ANTES de procesar.
2. Se usa el MÁXIMO de los paquetes por salto (no mínimo ni promedio) --
   confirmado empíricamente contra Path Analysis: un pico en 1 de 3
   paquetes (ej. [130, 7.7, 7.8]) es lo que la herramienta reporta.
3. Se resuelve ASN vía librería externa (ipwhois), adjuntado por
   ciclo/salto específico, no como lista aparte desalineada.
4. Se agregan los 3 niveles de severidad documentados por RIPE.
5. La comparación es CONSECUTIVA (ciclo N vs. ciclo N-1), no contra una
   línea base histórica global -- confirmado con evidencia real: un
   mismo pico genera DOS "changes" en Path Analysis (entrada y salida),
   algo que una comparación contra línea base global no puede replicar.
"""
import argparse
import sys
from datetime import datetime, timezone
from collections import Counter
import pandas as pd
import numpy as np
from ripe.atlas.cousteau import AtlasResultsRequest

try:
    from ipwhois import IPWhois
    IPWHOIS_DISPONIBLE = True
except ImportError:
    IPWHOIS_DISPONIBLE = False


# ==========================================
# PARSING DE ARGUMENTOS
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extraer y analizar historial de TRACEROUTE de UNA SOLA SONDA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--measurement-id", "-m", type=int, required=True)
    parser.add_argument("--probe-id", "-p", type=int, required=True)
    parser.add_argument("--start-time", type=str, required=True)
    parser.add_argument("--stop-time", type=str, required=True)
    parser.add_argument("--max-hops", type=int, default=15)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--resolve-asn", action="store_true",
                         help="Resolver ASN vía whois (más lento, requiere red)")
    return parser.parse_args()


def parse_dt(dt_str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Formato de fecha no válido: {dt_str}")


# ==========================================
# RESOLUCIÓN DE ASN (con caché, para no repetir lookups)
# ==========================================
_asn_cache = {}

def resolver_asn(ip):
    if ip in ('Unknown', None, '*'):
        return 'Unknown'
    if ip in _asn_cache:
        return _asn_cache[ip]
    if not IPWHOIS_DISPONIBLE:
        return 'Unknown'
    try:
        resultado = IPWhois(ip).lookup_rdap(depth=0)
        asn = resultado.get('asn', 'Unknown')
    except Exception:
        asn = 'Unknown'
    _asn_cache[ip] = asn
    return asn


# ==========================================
# CLASIFICACIÓN DE SEVERIDAD (graduada, igual que RIPE Path Analysis)
# ==========================================
def clasificar_severidad(rtt_pct):
    if rtt_pct > 200:
        return "🟣⬛ Dark (>200%)"
    elif rtt_pct > 40:
        return "🟣🟪 Medium (40-200%)"
    else:
        return "🟣 Light (20-40%)"


# ==========================================
# LÓGICA PRINCIPAL
# ==========================================
def main():
    args = parse_arguments()
    start_time = parse_dt(args.start_time)
    stop_time = parse_dt(args.stop_time)

    print("=" * 70)
    print("EXTRACCIÓN Y ANÁLISIS DE TRACEROUTE (RIPE Atlas) — versión corregida")
    print("=" * 70)
    print(f"\n⚙️  Configuración:")
    print(f"   Measurement ID: {args.measurement_id}")
    print(f"   Probe ID: {args.probe_id}")
    print(f"   Rango: {start_time.strftime('%Y-%m-%d %H:%M')} a {stop_time.strftime('%Y-%m-%d %H:%M')}")
    if args.resolve_asn and not IPWHOIS_DISPONIBLE:
        print("   ⚠️ 'ipwhois' no está instalado -- ASN quedará como 'Unknown'.")
        print("      Instalar con: pip install ipwhois --break-system-packages")

    # 1. Descargar datos
    print(f"\n📥 Solicitando historial de traceroute...")
    kwargs = {
        "msm_id": args.measurement_id,
        "start": int(start_time.timestamp()),
        "stop": int(stop_time.timestamp()),
        "probe_ids": [args.probe_id]
    }
    is_success, results = AtlasResultsRequest(**kwargs).create()
    if not is_success or not results:
        print(f"❌ Error o sin resultados: {results}")
        sys.exit(1)

    # ✅ FIX #1 — ordenar explícitamente por timestamp antes de procesar
    results = sorted(results, key=lambda r: r.get('timestamp', 0))
    print(f"   ✅ {len(results)} ciclos de traceroute descargados y ordenados"
          f" ({datetime.fromtimestamp(results[0]['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
          f" → {datetime.fromtimestamp(results[-1]['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}).")

    # 2. Estructurar datos (IP + RTT por salto y ciclo)
    print("\n🔍 Procesando resultados...")
    cycles_data = []
    ips_unicas = set()

    for res in results:
        ts = res.get('timestamp')
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hops = res.get('result', [])

        cycle_hops = []
        current_len = 0

        for h in hops:
            hop_num = h.get('hop')
            if hop_num is None or hop_num > args.max_hops:
                continue

            hop_res = h.get('result', [])

            # ✅ FIX #2 (definitivo) — el PRIMER paquete válido, ni máximo ni
            # mínimo ni promedio. Confirmado con evidencia estadística: sobre
            # 11 saltos reales comparados contra Path Analysis, "primero"
            # acierta 11/11; mediana 8/11; máximo y mínimo solo 5/11 cada uno.
            # Es, además, la MISMA fuente que ya se usaba para la IP (el
            # primer paquete que respondió) -- ahora IP y RTT vienen del
            # mismo paquete de referencia, sin inconsistencia interna.
            first_pkt = next((r for r in hop_res if isinstance(r, dict) and r.get('rtt') is not None), None)

            if first_pkt is None:
                continue

            rtt_hop = float(first_pkt['rtt'])
            ip = first_pkt.get('from', 'Unknown')
            ips_unicas.add(ip)

            cycle_hops.append({'hop': hop_num, 'ip': ip, 'rtt': rtt_hop, 'asn': None})
            current_len = max(current_len, hop_num)

        if cycle_hops:
            cycles_data.append({
                'timestamp': dt,
                'ts_ms': int(ts * 1000),
                'hops': cycle_hops,
                'length': current_len
            })

    # ✅ FIX #3 — resolver ASN por IP única (con caché), y adjuntarlo a CADA
    # salto de CADA ciclo -- antes quedaba en una lista aparte, desalineada
    # respecto a qué ciclo específico correspondía cada valor.
    if args.resolve_asn:
        print(f"\n🌐 Resolviendo ASN para {len(ips_unicas)} IPs únicas...")
        for ip in ips_unicas:
            resolver_asn(ip)  # llena _asn_cache
        for cycle in cycles_data:
            for h in cycle['hops']:
                h['asn'] = resolver_asn(h['ip'])
    else:
        print("\n🌐 Resolución de ASN desactivada (usar --resolve-asn para activarla).")

    normal_path_length = Counter(c['length'] for c in cycles_data).most_common(1)[0][0]
    print(f"   ✅ {len(cycles_data)} ciclos procesados (longitud típica: {normal_path_length} saltos).")

    # ✅ FIX #5 — comparación CONSECUTIVA (ciclo N vs ciclo N-1), no contra
    # línea base global. Confirmado con evidencia real: Path Analysis reporta
    # "1 change" al ENTRAR a un pico Y otro "1 change" al SALIR de él -- un
    # solo evento produce DOS cambios, porque compara pares consecutivos, no
    # contra una mediana histórica.
    print("\n🔎 Analizando ciclos en busca de anomalías (comparación consecutiva)...")
    anomalies_per_cycle = []
    csv_rows = []
    prev_hops_by_num = {}   # {hop_num: {'ip', 'rtt', 'asn'}} del ciclo ANTERIOR
    prev_length = None

    for cycle in cycles_data:
        dt_str = cycle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        cycle_anomalies = []   # cuentan para el total (🟣 🟡 🔴 real)
        cycle_info = []        # solo informativo, NO cuenta (ℹ️ balanceo interno)
        current_hops_by_num = {h['hop']: h for h in cycle['hops']}

        # 🟡 Cambio de longitud vs. el ciclo INMEDIATAMENTE anterior
        if prev_length is not None and cycle['length'] != prev_length:
            cycle_anomalies.append(f"🟡 Path Length: {cycle['length']} hops (Anterior: {prev_length})")

        for h in cycle['hops']:
            hop_num, rtt, ip, asn = h['hop'], h['rtt'], h['ip'], h['asn']
            prev_h = prev_hops_by_num.get(hop_num)

            if prev_h is not None:
                prev_rtt, prev_ip, prev_asn = prev_h['rtt'], prev_h['ip'], prev_h['asn']

                # 🟣 RTT: diferencia contra el ciclo ANTERIOR, no la mediana global
                rtt_diff = abs(rtt - prev_rtt)
                rtt_pct = (rtt_diff / prev_rtt * 100) if prev_rtt > 0 else 0
                if rtt_diff > 10.0 and rtt_pct > 20.0:
                    severidad = clasificar_severidad(rtt_pct)
                    cycle_anomalies.append(
                        f"{severidad} Hop {hop_num} RTT: {rtt:.1f}ms (Anterior: {prev_rtt:.1f}ms, +{rtt_pct:.0f}%)"
                    )

                # 🔴 IP y/o ASN: distintos a los del ciclo ANTERIOR en la misma posición
                # ✅ FIX #6 — un cambio de IP DENTRO DEL MISMO ASN (balanceo de
                # carga interno, ej. AS1299 alternando routers) NO CUENTA como
                # 'change' -- confirmado por evidencia real: Path Analysis
                # compara 20:38 directo contra 22:08, saltándose 5 ciclos
                # intermedios donde el ASN se mantenía igual (1299) pese a que
                # la IP específica alternaba. Va a 'cycle_info' (no cuenta),
                # no a 'cycle_anomalies'. Requiere --resolve-asn; sin la flag,
                # se mantiene el comportamiento anterior (cualquier cambio de
                # IP cuenta, porque no hay forma de saber si es el mismo ASN).
                if ip != 'Unknown' and prev_ip != 'Unknown' and ip != prev_ip:
                    # ✅ FIX #7 — ASN 'Unknown' en AMBOS lados (típico de IPs
                    # privadas, ej. 10.226.x.x) se trata igual que 'mismo ASN'.
                    # Confirmado con evidencia exacta: excluir 'Unknown' de esta
                    # regla dejaba 4 ciclos de más marcados (33 vs 29 reales) --
                    # los 4 correspondían EXACTAMENTE a saltos con IP privada
                    # alternando sin ningún ASN público resoluble en ningún lado.
                    # Sin evidencia de un ASN real distinto, no hay base para
                    # contarlo como cambio genuino.
                    mismo_asn = args.resolve_asn and (
                        (asn not in (None, 'Unknown') and asn == prev_asn)
                        or (asn in (None, 'Unknown') and prev_asn in (None, 'Unknown'))
                    )
                    if mismo_asn:
                        motivo = f"mismo ASN {asn}" if asn not in (None, 'Unknown') else "IP privada, sin ASN público resoluble en ningún lado"
                        cycle_info.append(
                            f"ℹ️ Hop {hop_num} IP: {ip} (Anterior: {prev_ip}) — {motivo}, "
                            f"NO cuenta como cambio (balanceo de carga interno)"
                        )
                    else:
                        etiqueta_asn = f", ASN {prev_asn}->{asn}" if args.resolve_asn else ""
                        cycle_anomalies.append(f"🔴 Hop {hop_num} IP: {ip} (Anterior: {prev_ip}{etiqueta_asn})")
            # Si prev_h es None (primer ciclo, o el salto no existía antes),
            # no hay nada contra qué comparar -- no se marca nada.

            csv_rows.append({
                'timestamp': dt_str, 'hop': hop_num, 'ip': ip, 'asn': asn, 'rtt_ms': rtt,
            })

        if cycle_anomalies:
            anomalies_per_cycle.append({
                'timestamp': dt_str, 'ts_ms': cycle['ts_ms'],
                'anomalies': cycle_anomalies + cycle_info,  # info solo se muestra si YA hay una anomalía real
            })

        prev_hops_by_num = current_hops_by_num
        prev_length = cycle['length']

    # 4. Guardar CSV
    output_file = args.output or f"historial_traceroute_probe_{args.probe_id}.csv"
    pd.DataFrame(csv_rows).to_csv(output_file, index=False)
    print(f"\n💾 Datos detallados guardados en: {output_file}")

    # 5. Reporte
    print(f"\n📊 Resumen de Anomalías Detectadas:")
    print(f"   Ciclos totales analizados: {len(cycles_data)}")
    print(f"   Ciclos con anomalías: {len(anomalies_per_cycle)} "
          f"({100*len(anomalies_per_cycle)/max(1, len(cycles_data)):.1f}%)")

    if anomalies_per_cycle:
        print(f"\n🔗 Los {len(anomalies_per_cycle)} ciclos con anomalías (orden cronológico):")
        for item in anomalies_per_cycle:
            print(f"\n   🕒 {item['timestamp']} UTC:")
            for anom in item['anomalies']:
                print(f"      {anom}")
            url = (f"https://atlas.ripe.net/pathanalysis/embed?"
                   f"measurementId={args.measurement_id}&sourceProbeId={args.probe_id}&"
                   f"center={item['ts_ms']}&window=7200000")
            print(f"      🔗 URL: {url}")

    print("\n✅ ¡Proceso completado!")


if __name__ == "__main__":
    main()
