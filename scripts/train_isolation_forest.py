#!/usr/bin/env python3
"""
train_isolation_forest.py — Fase 1: Isolation Forest por daypart (no supervisado)
═══════════════════════════════════════════════════════════════════════════════
Complementa (no reemplaza) al pipeline supervisado de train_anomaly_detection.py.
Objetivo: darle uso real a las capturas de link_health_monitor.py --mode real,
que hoy no pueden alimentar ningún pipeline supervisado (sin etiquetas).

✅ DISEÑO (ver conversación):
├─ UN modelo Isolation Forest POR daypart (8 buckets del esquema FCC — ver
│   link_health_feature_engine.py) — no un solo modelo global con 'daypart'
│   como feature. Esto es lo que resuelve "la normalidad depende de la
│   hora": el modelo de 03:00 aprende un perfil de variabilidad normal
│   distinto al de 20:00, sin fijar ningún umbral a mano.
├─ Features de entrada: SOLO las estadísticas relativas a la ventana móvil
│   (z_score, cv, p95_dev, z_deriv, trend_5min/15min, velocity,
│   acceleration, loss_spike) — deliberadamente NO se usa latency_ms/
│   jitter_ms/loss_pct crudos como eje principal, para no reintroducir por
│   la puerta trasera el mismo problema del umbral fijo que este pipeline
│   busca evitar.
└─ Validación: entrena SIN mirar la verdad de terreno (coherente con ser no
    supervisado), y solo la usa DESPUÉS para medir qué tan bien separa —
    reutilizando link_health_ground_truth (poblada únicamente en
    --mode synthetic) como arnés de validación indirecta, nunca como
    target de entrenamiento.

Requiere haber corrido, en este orden:
    1) migration_link_health.sql + migration_link_health_fcc_time.sql +
       migration_link_health_anomaly_scores.sql
    2) link_health_monitor.py --mode synthetic (para tener ground truth)
    3) link_health_feature_engine.py

USO:
    python3 train_isolation_forest.py
    python3 train_isolation_forest.py --days 60 --contamination 0.01
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from train_anomaly_detection import (
    load_long_format_dataset, CATEGORICAL_CONTEXT_FEATURES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ Deliberadamente SIN latency_ms/jitter_ms/loss_pct crudos — ver docstring.
RELATIVE_STAT_FEATURES = [
    'z_score', 'z_score_jitter', 'z_score_loss', 'cv', 'p95_dev', 'z_deriv',
    'trend_5min', 'trend_15min', 'velocity', 'acceleration', 'loss_spike',
]

DAYPART_ORDER = CATEGORICAL_CONTEXT_FEATURES['daypart']

MIN_SAMPLES_PER_DAYPART = 200  # por debajo de esto, no hay suficiente base para ajustar un bosque razonable


def prepare_relative_features(df):
    X = df[RELATIVE_STAT_FEATURES].copy()
    if 'loss_spike' in X.columns and X['loss_spike'].dtype == 'bool':
        X['loss_spike'] = X['loss_spike'].astype(int)
    X = X.fillna(0)
    object_cols = X.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    return X


def train_and_validate_one_daypart(df_bucket, daypart, contamination, test_size, random_state=42):
    """
    Entrena UN Isolation Forest sobre este daypart. Split train/test simple
    (no estratificado por label — el modelo no ve labels al ajustar, es no
    supervisado de verdad); el label solo se usa DESPUÉS, para validar.
    """
    if len(df_bucket) < MIN_SAMPLES_PER_DAYPART:
        logger.warning(f"⚠️ daypart '{daypart}': solo {len(df_bucket)} filas (mínimo {MIN_SAMPLES_PER_DAYPART}) — se omite")
        return None

    df_train, df_test = train_test_split(df_bucket, test_size=test_size, random_state=random_state, shuffle=True)

    X_train = prepare_relative_features(df_train)
    X_test = prepare_relative_features(df_test)
    y_test = df_test['is_anomaly'].astype(int).values

    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    model.fit(X_train)

    # sklearn: score_samples() más alto = más normal. Invertimos el signo
    # para que, en nuestras tablas/gráficos, "score alto" = "más anómalo"
    # (consistente con max_score/z_score del resto del proyecto).
    raw_scores = -model.score_samples(X_test)
    y_pred = (model.predict(X_test) == -1).astype(int)  # -1 = outlier en la convención sklearn

    metrics = {
        'n_test': len(df_test),
        'n_anomaly_real': int(y_test.sum()),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    # ROC-AUC necesita las dos clases presentes en el test set de este bucket
    if len(np.unique(y_test)) >= 2:
        metrics['roc_auc'] = roc_auc_score(y_test, raw_scores)
    else:
        metrics['roc_auc'] = np.nan

    return {'model': model, 'metrics': metrics}


def score_full_bucket(model, df_bucket, daypart):
    """
    Aplica el modelo YA ENTRENADO a TODAS las filas del bucket (train+test
    juntas, no solo el split de validación) — esto es lo que puebla
    link_health_anomaly_scores. Las métricas de desempeño (calculadas en
    train_and_validate_one_daypart, solo sobre el split de test) siguen
    siendo la estimación honesta de generalización; esto es la salida
    "operativa" completa, análoga a lo que haría el servicio de inferencia
    de Fase 3 sobre cada ciclo nuevo.
    """
    X_full = prepare_relative_features(df_bucket)
    raw_scores = -model.score_samples(X_full)
    is_anomaly = (model.predict(X_full) == -1)

    rows = []
    for (t, cyc, dns_t), score, flag in zip(
        df_bucket[['time', 'cycle_number', 'dns_target']].itertuples(index=False, name=None),
        raw_scores, is_anomaly
    ):
        rows.append((t, int(cyc), dns_t, daypart, float(score), bool(flag)))
    return rows


def write_scores(conn, rows):
    if not rows:
        return 0
    try:
        cur = conn.cursor()
        execute_values(
            cur,
            "INSERT INTO link_health_anomaly_scores "
            "(time, cycle_number, dns_target, daypart, isolation_forest_score, is_anomaly_unsupervised) VALUES %s",
            rows
        )
        conn.commit()
        cur.close()
        return len(rows)
    except Exception as e:
        logger.error(f"❌ Error insertando en link_health_anomaly_scores: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return 0


def main():
    parser = argparse.ArgumentParser(description='Entrena Isolation Forest por daypart (Fase 1, no supervisado)')
    parser.add_argument('--days', type=int, default=None, help='Límite de días hacia atrás (default: toda la tabla)')
    parser.add_argument('--contamination', type=str, default='auto',
                         help="Proporción esperada de outliers ('auto' o un float, ej. 0.01)")
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--models-dir', type=str, default='./models_isolation_forest',
                         help='Directorio donde persistir los modelos entrenados (uno por daypart)')
    parser.add_argument('--no-write-scores', action='store_true',
                         help='No escribir en link_health_anomaly_scores (por default SÍ se escribe)')
    parser.add_argument('--no-truncate', action='store_true',
                         help='No limpiar link_health_anomaly_scores antes de escribir (por default SÍ se limpia — '
                              'ver docstring: este script reprocesa todo el dataset en cada corrida)')
    parser.add_argument('--db-host', default='timescaledb')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', default='bgp_failover_db')
    parser.add_argument('--db-user', default='bgp_app')
    parser.add_argument('--db-password', default='bgp_app_password')
    args = parser.parse_args()

    contamination = args.contamination if args.contamination == 'auto' else float(args.contamination)

    print("=" * 80)
    print("🌲 ISOLATION FOREST POR DAYPART — Fase 1 (no supervisado, validado contra sintéticos)")
    print("=" * 80)

    print("\nPASO 1: Cargar dataset (mismo formato largo que train_anomaly_detection.py)")
    print("-" * 80)
    df = load_long_format_dataset(timescaledb_password='bgp_app_password', days=args.days)
    if df.empty:
        print("❌ Sin datos — abortando. ¿Corriste link_health_monitor.py --mode synthetic "
              "y link_health_feature_engine.py?")
        return

    print(f"✅ Dataset: {len(df)} filas | is_anomaly=True: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
    print(f"   (el modelo NO ve esta columna al entrenar — solo se usa acá abajo para validar)")

    os.makedirs(args.models_dir, exist_ok=True)

    write_conn = None
    if not args.no_write_scores:
        try:
            write_conn = psycopg2.connect(host=args.db_host, port=args.db_port, database=args.db_name,
                                           user=args.db_user, password=args.db_password)
            if not args.no_truncate:
                # ✅ Este script es de EXPERIMENTACIÓN OFFLINE (Fase 1) — cada
                # corrida reprocesa el dataset completo con la configuración
                # que le pasaste (contamination, --days, etc.). Sin este
                # TRUNCATE/DELETE, correrlo dos veces con distintos parámetros deja
                # ambas tandas de filas mezcladas en la misma tabla, sin forma
                # de distinguir cuál score vino de cuál corrida — la tabla no
                # tiene clave única que lo evite (ver migración). El futuro
                # servicio de inferencia continua (Fase 3) va a escribir de
                # forma distinta (solo ciclos NUEVOS, sin reprocesar todo), así
                # que este truncado es específico de este script, no del diseño
                # general de la tabla — usar --no-truncate si alguna vez hace
                # falta acumular en vez de reemplazar.
                cur = write_conn.cursor()
                cur.execute("DELETE FROM link_health_anomaly_scores")
                write_conn.commit()
                cur.close()
                logger.info("🧹 link_health_anomaly_scores limpiada antes de escribir esta corrida")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar/limpiar para escribir scores ({e}) — se continúa sin escribir")
            write_conn = None

    print("\nPASO 2: Entrenar y validar un Isolation Forest POR daypart")
    print("-" * 80)
    results = {}
    total_scores_written = 0
    for daypart in DAYPART_ORDER:
        df_bucket = df[df['daypart'] == daypart]
        result = train_and_validate_one_daypart(df_bucket, daypart, contamination, args.test_size)
        if result is None:
            continue
        results[daypart] = result

        model_path = os.path.join(args.models_dir, f"isolation_forest_{daypart.replace(':', '')}.joblib")
        joblib.dump(result['model'], model_path)

        m = result['metrics']
        print(f"  {daypart:8s} | n_test={m['n_test']:5d} | anomalías reales={m['n_anomaly_real']:3d} | "
              f"Precision={m['precision_macro']:.3f} | Recall={m['recall_macro']:.3f} | "
              f"F1={m['f1_macro']:.3f} | ROC-AUC={m['roc_auc']:.3f} | modelo -> {model_path}")

        if write_conn is not None:
            score_rows = score_full_bucket(result['model'], df_bucket, daypart)
            written = write_scores(write_conn, score_rows)
            total_scores_written += written

    if write_conn is not None:
        print(f"\n💾 {total_scores_written} filas escritas en link_health_anomaly_scores "
              f"(todo el bucket de cada daypart entrenado, no solo el split de test)")
        write_conn.close()
    elif args.no_write_scores:
        print("\nℹ️ --no-write-scores: no se escribió nada en link_health_anomaly_scores")

    if not results:
        print("\n❌ Ningún daypart tuvo suficientes datos — abortando resumen.")
        return

    print("\n" + "=" * 80)
    print("✅ RESUMEN")
    print("=" * 80)
    all_metrics = pd.DataFrame([r['metrics'] for r in results.values()], index=results.keys())

    # ⚠️ Separación IMPORTANTE: los dayparts sin ninguna anomalía real en el
    # test set (ej. horario valle en escala 'realistic', donde el generador
    # nunca inyecta episodios) no miden "el modelo falla en detectar" —
    # miden un artefacto de que IsolationForest(contamination=...) está
    # forzado a marcar una fracción como outlier SIEMPRE, haya o no algo
    # anómalo de verdad. Meterlos en el promedio agregado diluiría
    # injustamente el desempeño real en los buckets donde sí hay algo que
    # detectar. Se reportan aparte, como "sin cobertura de validación", no
    # como mal desempeño.
    covered = all_metrics[all_metrics['n_anomaly_real'] > 0]
    uncovered = all_metrics[all_metrics['n_anomaly_real'] == 0]

    if len(uncovered) > 0:
        print(f"\n⚠️ {len(uncovered)} daypart(s) SIN anomalías reales en el test set "
              f"({', '.join(uncovered.index)}) — sin cobertura de validación en esta corrida,")
        print("   NO se incluyen en el promedio de abajo (dragarían el resultado por un artefacto")
        print("   de contamination, no por desempeño real). Esperable en --scale realistic, donde")
        print("   el generador solo inyecta episodios en horario pico.")

    if len(covered) == 0:
        print("\n❌ Ningún daypart tuvo anomalías reales en el test set — no se puede resumir desempeño.")
        return

    print(f"\n📊 Promedio macro ENTRE los {len(covered)} dayparts CON cobertura de validación "
          f"(cada bucket pesa igual, sin importar su tamaño):")
    for col in ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']:
        print(f"  {col:18s}: {covered[col].mean():.4f} ± {covered[col].std():.4f}")

    print(f"\n📁 {len(results)} modelos persistidos en: {args.models_dir}/")
    print("   (uno por daypart — reutilizables por el futuro servicio de inferencia continua, Fase 3)")

    print("\n⚠️ Lectura: este pipeline es NO SUPERVISADO — no ve is_anomaly al entrenar.")
    print("   Las métricas de arriba miden qué tan bien el modelo, sin etiquetas, se aproxima")
    print("   a la verdad de terreno sintética conocida — no reemplazan al pipeline supervisado")
    print("   de train_anomaly_detection.py, lo complementan para el caso sin etiquetas (--mode real).")
    print("=" * 80)


if __name__ == '__main__':
    main()
