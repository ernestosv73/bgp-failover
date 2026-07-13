#!/usr/bin/env python3
"""
link_health_monitor.py — Captura de salud de un enlace BGP hacia DNS1/DNS2
═══════════════════════════════════════════════════════════════════════════════
Pipeline INDEPENDIENTE del motor de failover (bgp_failover_engine_new.py).
No hay lógica dual-provider, no hay scoring, no hay decisión de ruta — solo
captura continua de latencia/jitter/pérdida hacia DNS1/DNS2, a través de UN
enlace con un provider principal (el hop del peer queda incluido
implícitamente en la medición end-to-end, no se monitorea por separado —
ver conversación).

Objetivo del pipeline completo: entrenar un modelo que ESTIME umbrales de
latencia y ciclos de degradación sostenida a partir de los datos, en vez de
asumirlos de antemano (ver train_anomaly_detection.py).

✅ DOS MODOS:
├─ --mode real: ejecuta MTR de verdad contra DNS1/DNS2 (misma lógica que
│   bgp_failover_engine_new.py, simplificada — ya no hace falta el fallback
│   de peer ni el manejo dual-hop, solo se toma el hop que coincide con la
│   IP de destino). Cadencia real (sleep entre ciclos).
│   ⚠️ En este modo NO se puebla link_health_ground_truth — no hay forma
│   automática de saber si un ciclo de tráfico real es o no una anomalía.
│   Si se inyecta latencia real (ej. tc qdisc netem) y se quiere entrenar
│   con eso, hay que etiquetar manualmente (ver conversación).
├─ --mode synthetic: reutiliza EXACTAMENTE la misma ScenarioGenerator (5
│   tipos de episodio: step/spike/unstable/oscillating/slow_increase) ya
│   validada en synthetic_data_generator.py — importada de ahí, no
│   duplicada. A diferencia de aquel script, acá NO hay que envolverla en
│   ningún motor de failover (BGPFailoverEngine): se llama a
│   scenario.next_metrics() directo en un loop simple, y se descarta
│   peer_avg/peer_stddev (no se monitorea el peer en este pipeline).
│   Puebla AMBAS tablas: link_health_metrics y link_health_ground_truth.

USO:
    # Modo real (contra la topología Containerlab)
    python3 link_health_monitor.py --mode real --cycles 500

    # Modo sintético, escala rápida de laboratorio
    python3 link_health_monitor.py --mode synthetic --scale lab --cycles 5000

    # Modo sintético, escala calibrada al ISP
    python3 link_health_monitor.py --mode synthetic --scale realistic --cycles 100000
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import json
import logging
import random
import subprocess
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# ✅ Reutiliza la máquina de estados de episodios ya validada — no se
# duplica. Este script NO importa nada del motor de failover.
from synthetic_data_generator import (
    ScenarioGenerator, SCALE_PRESETS, compute_calibrated_probability,
    REAL_CYCLE_INTERVAL_SECONDS, DEFAULT_OU_THETA, SUSTAINED_EPISODE_TYPES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Configuración de captura real (MTR) ─────────────────────────────────
MTR_CONFIG = {'count': 5, 'timeout': 30, 'packet_size': 64, 'interval': 0.5}
IP_VERSION = '6'
DNS_DESTINATIONS = {
    'dns1': '2001:db8:8888::100',
    'dns2': '2001:db8:4444::100',
}

GROUND_TRUTH_FLUSH_EVERY_CYCLES = 500


# ════════════════════════════════════════════════════════════════════════
# DB helpers
# ════════════════════════════════════════════════════════════════════════

def get_connection(host, port, db, user, password):
    return psycopg2.connect(host=host, port=port, database=db, user=user, password=password)


def get_last_cycle_number(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(cycle_number), 0) FROM link_health_metrics")
        result = cur.fetchone()
        cur.close()
        return int(result[0]) if result else 0
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer cycle_number previo (¿corriste migration_link_health.sql?): {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return 0


def get_last_timestamp(conn):
    """
    Último 'time' almacenado en link_health_metrics, o None si la tabla
    está vacía. Se usa para continuar la línea de tiempo simulada de forma
    consecutiva entre corridas — sin esto, cycle_number continúa
    correctamente pero el timestamp vuelve a arrancar desde "ahora" cada
    vez, dejando huecos o solapamientos en el rango de tiempo simulado.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(time) FROM link_health_metrics")
        result = cur.fetchone()
        cur.close()
        return result[0] if result and result[0] is not None else None
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer el último timestamp: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def insert_metrics_row(conn, row):
    try:
        cur = conn.cursor()
        cols = list(row.keys())
        placeholders = ', '.join(['%s'] * len(cols))
        cur.execute(
            f"INSERT INTO link_health_metrics ({', '.join(cols)}) VALUES ({placeholders})",
            [row[c] for c in cols]
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.error(f"❌ Error insertando en link_health_metrics: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def insert_ground_truth_batch(conn, rows):
    if not rows:
        return 0
    try:
        cur = conn.cursor()
        execute_values(
            cur,
            "INSERT INTO link_health_ground_truth "
            "(time, cycle_number, dns_target, episode_type, is_anomaly, is_sustained) VALUES %s",
            rows
        )
        conn.commit()
        cur.close()
        return len(rows)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo insertar en link_health_ground_truth: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None


# ════════════════════════════════════════════════════════════════════════
# Modo REAL — MTR contra la topología
# ════════════════════════════════════════════════════════════════════════

def run_mtr(destination):
    try:
        cmd = [
            'mtr', f'-{IP_VERSION}', '-n', '-j',
            '-c', str(MTR_CONFIG['count']),
            '-s', str(MTR_CONFIG['packet_size']),
            '-i', str(MTR_CONFIG['interval']),
            destination,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=MTR_CONFIG['timeout'])
        if result.returncode == 0:
            return json.loads(result.stdout)
        logger.warning(f"⚠️ MTR a {destination} falló (código {result.returncode}): {result.stderr}")
        return None
    except Exception as e:
        logger.error(f"❌ Error ejecutando MTR a {destination}: {e}")
        return None


def extract_last_hop(mtr_report, destination_ip):
    """Extrae las métricas del hop que coincide con la IP de destino — sin
    fallback de peer, sin lógica dual-hop. Solo nos interesa la medición
    end-to-end hacia el DNS."""
    try:
        hubs = mtr_report['report']['hubs']
        for hop in hubs:
            if hop.get('host') == destination_ip:
                return {
                    'avg': float(hop.get('Avg', float('inf'))),
                    'loss': float(hop.get('Loss%', 0.0)),
                    'stddev': float(hop.get('StDev', 0.0)),
                }
        logger.warning(f"⚠️ No se encontró el hop de destino ({destination_ip}) en la traza MTR")
        return None
    except Exception as e:
        logger.error(f"❌ Error extrayendo métricas MTR: {e}")
        return None


def run_real_capture(conn, cycles, interval_seconds):
    cycle_number = get_last_cycle_number(conn) + 1
    logger.info("=" * 80)
    logger.info(f"📡 Captura REAL — {cycles} ciclos, cadencia objetivo {interval_seconds}s")
    logger.info(f"   DNS1: {DNS_DESTINATIONS['dns1']} | DNS2: {DNS_DESTINATIONS['dns2']}")
    logger.info(f"   cycle_number inicial (continúa desde BD): {cycle_number}")
    logger.info("=" * 80)

    ok, failed = 0, 0
    for i in range(cycles):
        t0 = time.time()

        mtr1 = run_mtr(DNS_DESTINATIONS['dns1'])
        m1 = extract_last_hop(mtr1, DNS_DESTINATIONS['dns1']) if mtr1 else None
        mtr2 = run_mtr(DNS_DESTINATIONS['dns2'])
        m2 = extract_last_hop(mtr2, DNS_DESTINATIONS['dns2']) if mtr2 else None

        if m1 is None or m2 is None:
            logger.warning(f"⚠️ Ciclo {cycle_number} incompleto — se omite (no se inserta fila parcial)")
            failed += 1
            cycle_number += 1
            elapsed = time.time() - t0
            time.sleep(max(0, interval_seconds - elapsed))
            continue

        row = {
            'time': datetime.now(timezone.utc), 'cycle_number': cycle_number,
            'dns1_latency_ms': m1['avg'], 'dns1_jitter_ms': m1['stddev'], 'dns1_loss_pct': m1['loss'],
            'dns2_latency_ms': m2['avg'], 'dns2_jitter_ms': m2['stddev'], 'dns2_loss_pct': m2['loss'],
        }
        if insert_metrics_row(conn, row):
            ok += 1
        logger.info(f"✓ Ciclo {cycle_number}: DNS1={m1['avg']:.2f}ms (loss {m1['loss']:.1f}%) | "
                     f"DNS2={m2['avg']:.2f}ms (loss {m2['loss']:.1f}%)")

        cycle_number += 1
        elapsed = time.time() - t0
        time.sleep(max(0, interval_seconds - elapsed))

    logger.info("=" * 80)
    logger.info(f"✅ Captura real completa: {ok} ciclos insertados, {failed} omitidos por error de MTR")
    logger.info(f"   ⚠️ link_health_ground_truth NO se pobló (modo real) — ver docstring del módulo "
                f"si necesitás entrenar con estos datos.")
    logger.info("=" * 80)


# ════════════════════════════════════════════════════════════════════════
# Modo SYNTHETIC — reutiliza ScenarioGenerator
# ════════════════════════════════════════════════════════════════════════

def run_synthetic_capture(conn, cycles, scale, seed, start_time, interval_seconds,
                           events_per_week, peak_hour_start, peak_hour_duration,
                           confirmation_cycles, episode_probability_override,
                           episode_type_weights, noise_model, ou_theta):
    preset = SCALE_PRESETS[scale]
    resolved_confirmation_cycles = confirmation_cycles or preset['confirmation_cycles']
    resolved_events_per_week = events_per_week if events_per_week is not None else preset['events_per_week']
    peak_hour_only = preset['peak_hour_only']

    if episode_probability_override is not None:
        resolved_probability = episode_probability_override
    elif resolved_events_per_week is not None:
        resolved_probability = compute_calibrated_probability(
            resolved_events_per_week, interval_seconds, peak_hour_duration)
    else:
        resolved_probability = preset['episode_probability']

    scenario = ScenarioGenerator(
        episode_probability=resolved_probability,
        confirmation_cycles=resolved_confirmation_cycles,
        peak_hour_only=peak_hour_only,
        peak_hour_start=peak_hour_start,
        peak_hour_duration=peak_hour_duration,
        episode_type_weights=episode_type_weights,
        noise_model=noise_model,
        ou_theta=ou_theta,
        seed=seed,
    )

    cycle_number = get_last_cycle_number(conn) + 1
    logger.info("=" * 80)
    logger.info(f"🎲 Captura SINTÉTICA — {cycles} ciclos, escala '{scale}'")
    logger.info(f"   Ventana de degradación sostenida: {resolved_confirmation_cycles} ciclos "
                f"(~{resolved_confirmation_cycles * interval_seconds / 60:.1f} min a {interval_seconds}s/ciclo)")
    logger.info(f"   Mezcla de tipos de episodio: {scenario.episode_type_weights}")
    logger.info(f"   Modelo de ruido de fondo: {noise_model}"
                + (f" (theta={ou_theta:.3f})" if noise_model == 'ou' else ""))
    logger.info(f"   cycle_number inicial (continúa desde BD): {cycle_number}")
    logger.info("=" * 80)

    ground_truth_buffer = []
    ground_truth_inserted = 0
    ground_truth_failed = False
    current_time = start_time

    for i in range(cycles):
        scenario.set_current_time(current_time)
        raw = scenario.next_metrics()  # peer_avg/peer_stddev se descartan — no se monitorea peer acá

        row = {
            'time': current_time, 'cycle_number': cycle_number,
            'dns1_latency_ms': raw['dns1_avg'], 'dns1_jitter_ms': raw['dns1_stddev'], 'dns1_loss_pct': 0.0,
            'dns2_latency_ms': raw['dns2_avg'], 'dns2_jitter_ms': raw['dns2_stddev'], 'dns2_loss_pct': 0.0,
        }
        insert_metrics_row(conn, row)

        if not ground_truth_failed:
            active_target = scenario.last_active_target
            active_type = scenario.last_active_episode_type
            for dns_name in ('dns1', 'dns2'):
                is_anomaly = (active_target == dns_name)
                ep_type = active_type if is_anomaly else None
                ground_truth_buffer.append((
                    current_time, cycle_number, dns_name, ep_type,
                    is_anomaly, bool(is_anomaly and ep_type in SUSTAINED_EPISODE_TYPES),
                ))

        if len(ground_truth_buffer) >= GROUND_TRUTH_FLUSH_EVERY_CYCLES * 2:
            inserted = insert_ground_truth_batch(conn, ground_truth_buffer)
            if inserted is None:
                ground_truth_failed = True
                logger.warning("⚠️ Se desactivó el registro de verdad de terreno para el resto de esta corrida "
                                "(¿corriste migration_link_health.sql?)")
            else:
                ground_truth_inserted += inserted
            ground_truth_buffer = []

        cycle_number += 1
        current_time = current_time + timedelta(seconds=interval_seconds)

        if (i + 1) % max(1, cycles // 20) == 0:
            logger.info(f"   ✓ {i + 1}/{cycles} ciclos generados")

    if not ground_truth_failed and ground_truth_buffer:
        inserted = insert_ground_truth_batch(conn, ground_truth_buffer)
        if inserted:
            ground_truth_inserted += inserted

    logger.info("=" * 80)
    logger.info(f"✅ Captura sintética completa: {cycles} ciclos insertados en link_health_metrics")
    if not ground_truth_failed:
        logger.info(f"✅ Verdad de terreno: {ground_truth_inserted} filas en link_health_ground_truth")
    logger.info(f"   Rango simulado: {start_time.isoformat()} → {current_time.isoformat()}")
    logger.info("=" * 80)


# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Captura de salud de enlace BGP hacia DNS1/DNS2 (real o sintética)')
    parser.add_argument('--mode', choices=['real', 'synthetic'], required=True)
    parser.add_argument('--cycles', type=int, default=300)
    parser.add_argument('--interval-seconds', type=int, default=REAL_CYCLE_INTERVAL_SECONDS)

    # DB
    parser.add_argument('--db-host', default='timescaledb')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', default='bgp_failover_db')
    parser.add_argument('--db-user', default='bgp_app')
    parser.add_argument('--db-password', default='bgp_app_password')

    # Solo modo synthetic
    parser.add_argument('--scale', choices=['lab', 'realistic'], default='lab')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--days-back', type=float, default=None)
    parser.add_argument('--events-per-week', type=float, default=None)
    parser.add_argument('--peak-hour-start', type=int, default=19)
    parser.add_argument('--peak-hour-duration', type=int, default=4)
    parser.add_argument('--confirmation-cycles', type=int, default=None,
                         help='Ventana de degradación sostenida, en ciclos (default del preset)')
    parser.add_argument('--episode-probability', type=float, default=None)
    parser.add_argument('--episode-type-weights', type=str, default=None,
                         help='JSON, ej. \'{"step":0.45,"spike":0.15,"unstable":0.15,"oscillating":0.10,"slow_increase":0.15}\'')
    parser.add_argument('--noise-model', choices=['iid', 'ou'], default='ou')
    parser.add_argument('--ou-theta', type=float, default=DEFAULT_OU_THETA)

    args = parser.parse_args()

    conn = get_connection(args.db_host, args.db_port, args.db_name, args.db_user, args.db_password)

    if args.mode == 'real':
        run_real_capture(conn, args.cycles, args.interval_seconds)
    else:
        random.seed(args.seed)

        last_timestamp = get_last_timestamp(conn)

        if args.days_back is not None:
            # Override explícito del usuario — se respeta, pero se avisa si
            # deja un hueco o un solapamiento con lo ya almacenado.
            start_time = datetime.now(timezone.utc) - timedelta(days=args.days_back)
            if last_timestamp is not None:
                expected_next = last_timestamp + timedelta(seconds=args.interval_seconds)
                if abs((start_time - expected_next).total_seconds()) > args.interval_seconds:
                    gap_or_overlap = "hueco" if start_time > expected_next else "solapamiento"
                    logger.warning(
                        f"⚠️ --days-back fue especificado explícitamente: la nueva corrida va a generar un "
                        f"{gap_or_overlap} en la línea de tiempo respecto a lo ya almacenado. "
                        f"Último timestamp en BD: {last_timestamp.isoformat()} | "
                        f"Nuevo inicio: {start_time.isoformat()}"
                    )
        elif last_timestamp is not None:
            # ✅ Continuación consecutiva: igual que cycle_number, el timestamp
            # arranca justo después del último ciclo ya almacenado — no desde
            # "ahora". Esto es lo que permite generar un dataset cada vez más
            # grande corriendo el script varias veces SIN truncar la tabla.
            start_time = last_timestamp + timedelta(seconds=args.interval_seconds)
            logger.info(f"⏩ Continuando línea de tiempo desde el último timestamp almacenado: "
                        f"{last_timestamp.isoformat()} -> arranca en {start_time.isoformat()}")
        else:
            # Cold start: tabla vacía, no hay nada de qué continuar.
            start_time = datetime.now(timezone.utc) - timedelta(seconds=args.interval_seconds * args.cycles)

        episode_type_weights = None
        if args.episode_type_weights:
            episode_type_weights = json.loads(args.episode_type_weights)

        run_synthetic_capture(
            conn, args.cycles, args.scale, args.seed, start_time, args.interval_seconds,
            args.events_per_week, args.peak_hour_start, args.peak_hour_duration,
            args.confirmation_cycles, args.episode_probability, episode_type_weights,
            args.noise_model, args.ou_theta,
        )

    conn.close()


if __name__ == '__main__':
    main()
