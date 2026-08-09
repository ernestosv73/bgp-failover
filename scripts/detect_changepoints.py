#!/usr/bin/env python3
"""
detect_changepoints.py — Fase 2: detección de puntos de cambio (PELT)
═══════════════════════════════════════════════════════════════════════════════
Complementa a train_isolation_forest.py (Fase 1). Diferencia clave, deliberada:

  Isolation Forest (Fase 1)  necesita link_health_features (derivadas) para
                              funcionar — depende de link_health_feature_engine.py.
  PELT (Fase 2, este script) opera DIRECTO sobre link_health_metrics (la
                              serie cruda de latencia) — no necesita features
                              derivadas NI verdad de terreno para correr.

Esto es intencional: el objetivo de este pipeline es que un ISP pueda usarlo
sobre captura real (--mode real) con el mínimo de pasos intermedios. Acá se
valida contra datos sintéticos como chequeo de cordura antes de confiar en
el método — no porque el script LO NECESITE para funcionar.

✅ QUÉ HACE:
Para cada dns_target (dns1, dns2) por separado, corre PELT sobre la serie
temporal de latencia cruda para encontrar los instantes donde cambió el
régimen estadístico (media y/o varianza) de forma sostenida — a diferencia
de Isolation Forest, que puntúa CADA ciclo por separado, PELT mira el
SEGMENTO completo y encuentra el quiebre.

Por cada punto de cambio: calcula media/desvío del segmento ANTES y
DESPUÉS, y los correlaciona automáticamente contra los umbrales de
referencia (REFERENCE_THRESHOLDS_MS, importados de train_anomaly_detection.py)
— exactamente la pieza que motivó proponer PELT en primer lugar.

Requiere haber corrido:
    1) migration_link_health.sql + migration_link_health_changepoints.sql
    2) link_health_monitor.py (--mode real O --mode synthetic — PELT no
       distingue, ambos poblan link_health_metrics de la misma forma)

USO:
    python3 detect_changepoints.py
    python3 detect_changepoints.py --days 30 --penalty 200
    python3 detect_changepoints.py --validate-against-synthetic   # solo si hay link_health_ground_truth
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import logging

import numpy as np
import pandas as pd
import psycopg2
import ruptures as rpt
from psycopg2.extras import execute_values

from train_anomaly_detection import REFERENCE_THRESHOLDS_MS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ventana usada para calcular media/desvío a cada lado del punto de cambio
# (misma convención que TREND_WINDOW_LONG en el resto del proyecto).
SEGMENT_STATS_WINDOW = 30

DNS_COLUMNS = {'dns1': 'dns1_latency_ms', 'dns2': 'dns2_latency_ms'}


def load_raw_series(conn, days=None):
    """Carga DIRECTO de link_health_metrics — sin pasar por el feature engine."""
    time_filter = f"WHERE time >= NOW() - INTERVAL '{days} days'" if days is not None else ""
    query = f"""
        SELECT time, cycle_number, dns1_latency_ms, dns2_latency_ms
        FROM link_health_metrics
        {time_filter}
        ORDER BY time
    """
    df = pd.read_sql(query, conn)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df


def detect_changepoints_for_series(df, dns_name, model='rbf', penalty=10, min_size=5):
    """
    Corre PELT sobre la serie de un solo dns_target. Devuelve una lista de
    índices (posiciones en df) donde PELT ubicó un quiebre — excluye el
    último breakpoint trivial que ruptures siempre agrega (fin de la serie).
    """
    col = DNS_COLUMNS[dns_name]
    signal = df[col].ffill().bfill().values.reshape(-1, 1)

    if len(signal) < min_size * 2:
        logger.warning(f"⚠️ {dns_name}: serie muy corta ({len(signal)} puntos) para PELT — se omite")
        return []

    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    breakpoints = algo.predict(pen=penalty)
    return breakpoints[:-1]  # el último es siempre len(signal), no es un quiebre real


def build_changepoint_rows(df, dns_name, breakpoints):
    col = DNS_COLUMNS[dns_name]
    thresholds = REFERENCE_THRESHOLDS_MS[dns_name]
    rows = []

    for bp in breakpoints:
        before = df[col].iloc[max(0, bp - SEGMENT_STATS_WINDOW):bp]
        after = df[col].iloc[bp:bp + SEGMENT_STATS_WINDOW]
        if len(before) == 0 or len(after) == 0:
            continue

        before_mean, before_std = float(before.mean()), float(before.std())
        after_mean, after_std = float(after.mean()), float(after.std())

        crosses_warning = (after_mean >= thresholds['warning']) and (before_mean < thresholds['warning'])
        crosses_critical = (after_mean >= thresholds['critical']) and (before_mean < thresholds['critical'])

        change_time = df['time'].iloc[bp] if bp < len(df) else df['time'].iloc[-1]
        rows.append((
            dns_name, change_time, before_mean, before_std, after_mean, after_std,
            bool(crosses_warning), bool(crosses_critical),
        ))
    return rows


def write_changepoints(conn, rows):
    if not rows:
        return 0
    cur = conn.cursor()
    execute_values(
        cur,
        "INSERT INTO link_health_changepoints "
        "(dns_target, change_time, segment_before_mean, segment_before_std, "
        "segment_after_mean, segment_after_std, crosses_warning_threshold, crosses_critical_threshold) VALUES %s",
        rows
    )
    conn.commit()
    cur.close()
    return len(rows)


def validate_against_synthetic(conn, dns_name, breakpoints, df, tolerance_cycles=10):
    """
    Chequeo de cordura OPCIONAL — solo tiene sentido si hay
    link_health_ground_truth poblada (--mode synthetic). Mide, de los
    inicios reales de episodios SOSTENIDOS (step/slow_increase — los que
    PELT debería poder encontrar, por diseño), qué fracción tiene un punto
    de cambio detectado a menos de tolerance_cycles de distancia.
    """
    try:
        gt_query = f"""
            SELECT cycle_number, is_sustained, episode_type
            FROM link_health_ground_truth
            WHERE dns_target = '{dns_name}'
            ORDER BY cycle_number
        """
        gt = pd.read_sql(gt_query, conn)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo validar contra sintéticos (¿corriste --mode synthetic?): {e}")
        return None

    if gt.empty:
        return None

    gt = gt.merge(df[['cycle_number']].reset_index().rename(columns={'index': 'row_idx'}), on='cycle_number', how='inner')
    gt_sorted = gt.sort_values('cycle_number').reset_index(drop=True)
    onsets = gt_sorted[
        gt_sorted['is_sustained'] & ~gt_sorted['is_sustained'].shift(1, fill_value=False)
    ]['row_idx'].tolist()

    if not onsets:
        return None

    detected = 0
    for onset_idx in onsets:
        if any(abs(bp - onset_idx) <= tolerance_cycles for bp in breakpoints):
            detected += 1

    return {'n_onsets_reales': len(onsets), 'n_detectados': detected,
            'tasa_deteccion': detected / len(onsets)}


def main():
    parser = argparse.ArgumentParser(description='Detección de puntos de cambio (PELT) — Fase 2, opera sobre link_health_metrics crudo')
    parser.add_argument('--days', type=int, default=None, help='Límite de días hacia atrás (default: toda la tabla)')
    parser.add_argument('--model', choices=['rbf', 'l2', 'l1'], default='l2',
                         help="Modelo de costo de PELT (default: l2). ⚠️ 'rbf' construye una matriz "
                              "kernel N×N — con series de decenas de miles de ciclos, esto puede agotar "
                              "la memoria (probado: 20.000 ciclos con rbf -> proceso terminado por el SO). "
                              "'l2' detecta cambios de MEDIA (cubre step/slow_increase) con complejidad "
                              "lineal, escala sin problema. Usar 'rbf' solo con --days chico.")
    parser.add_argument('--penalty', type=float, default=100,
                         help='Penalización de PELT (default: 100 — calibrado empíricamente: valores '
                              'bajos como 10 son extremadamente sensibles a ruido normal, ej. miles de '
                              '"quiebres" espurios en 20.000 ciclos. 100 mantuvo 100%% de detección de '
                              'inicios reales en la validación de referencia, con un conteo mucho más '
                              'razonable. dns2 suele necesitar penalty más alto que dns1 para el mismo '
                              'nivel de precisión — mismo patrón que la calibración por bucket en Fase 1.')
    parser.add_argument('--min-size', type=int, default=5, help='Mínimo de ciclos por segmento (default: 5)')
    parser.add_argument('--validate-against-synthetic', action='store_true',
                         help='Chequeo de cordura opcional contra link_health_ground_truth (requiere --mode synthetic)')
    parser.add_argument('--db-host', default='timescaledb')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', default='bgp_failover_db')
    parser.add_argument('--db-user', default='bgp_app')
    parser.add_argument('--db-password', default='bgp_app_password')
    parser.add_argument('--no-write', action='store_true', help='No escribir en link_health_changepoints')
    args = parser.parse_args()

    print("=" * 80)
    print("📈 DETECCIÓN DE PUNTOS DE CAMBIO (PELT) — Fase 2, opera sobre datos crudos")
    print("=" * 80)
    print(f"\n📚 Umbrales de referencia: DNS1 warning={REFERENCE_THRESHOLDS_MS['dns1']['warning']}ms "
          f"critical={REFERENCE_THRESHOLDS_MS['dns1']['critical']}ms | "
          f"DNS2 warning={REFERENCE_THRESHOLDS_MS['dns2']['warning']}ms "
          f"critical={REFERENCE_THRESHOLDS_MS['dns2']['critical']}ms")

    conn = psycopg2.connect(host=args.db_host, port=args.db_port, database=args.db_name,
                             user=args.db_user, password=args.db_password)

    print("\nPASO 1: Cargar serie cruda desde link_health_metrics (sin pasar por el feature engine)")
    print("-" * 80)
    df = load_raw_series(conn, days=args.days)
    if df.empty:
        print("❌ Sin datos en link_health_metrics — abortando. ¿Corriste link_health_monitor.py?")
        conn.close()
        return
    print(f"✅ {len(df)} ciclos cargados ({df['time'].min()} → {df['time'].max()})")

    if not args.no_write:
        cur = conn.cursor()
        cur.execute("DELETE FROM link_health_changepoints")
        conn.commit()
        cur.close()
        logger.info("🧹 link_health_changepoints limpiada antes de escribir esta corrida")

    print(f"\nPASO 2: Correr PELT (model={args.model}, penalty={args.penalty}, min_size={args.min_size})")
    print("-" * 80)
    total_written = 0
    for dns_name in ('dns1', 'dns2'):
        breakpoints = detect_changepoints_for_series(df, dns_name, model=args.model,
                                                       penalty=args.penalty, min_size=args.min_size)
        print(f"\n  {dns_name}: {len(breakpoints)} puntos de cambio detectados")

        rows = build_changepoint_rows(df, dns_name, breakpoints)
        n_warning = sum(1 for r in rows if r[6])
        n_critical = sum(1 for r in rows if r[7])
        print(f"     -> {n_warning} cruzan warning | {n_critical} cruzan critical")

        if not args.no_write:
            written = write_changepoints(conn, rows)
            total_written += written
            print(f"     -> {written} filas escritas en link_health_changepoints")

        if args.validate_against_synthetic:
            result = validate_against_synthetic(conn, dns_name, breakpoints, df)
            if result:
                print(f"     🔍 Chequeo de cordura: {result['n_detectados']}/{result['n_onsets_reales']} "
                      f"inicios de episodio sostenido detectados dentro de ±10 ciclos "
                      f"({result['tasa_deteccion']*100:.1f}%)")
            else:
                print(f"     🔍 Sin verdad de terreno disponible para validar (¿--mode synthetic?)")

    print("\n" + "=" * 80)
    print(f"✅ Total: {total_written} puntos de cambio persistidos" if not args.no_write else "✅ Corrida completa (sin escribir)")
    print("=" * 80)
    print("\n⚠️ Lectura: PELT no necesita features derivadas ni verdad de terreno para correr —")
    print("   funciona igual sobre captura real (--mode real) que sobre datos sintéticos. La")
    print("   validación contra sintéticos (--validate-against-synthetic) es solo un chequeo de")
    print("   cordura antes de confiar en el método, no un requisito de funcionamiento.")
    print("=" * 80)

    conn.close()


if __name__ == '__main__':
    main()
