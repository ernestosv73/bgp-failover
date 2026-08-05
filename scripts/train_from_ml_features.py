#!/usr/bin/env python3
"""
train_from_ml_features.py — v2: carga real desde ml_features (Opción A, multiclase)
✅ REESCRITO COMPLETO:
├─ Query alineada al schema REAL de ml_features (post-migración v2) — ya no
│   referencia columnas eliminadas (latency_ratio, quality_index,
│   score_difference, absolute_severity, etc.)
├─ Target: target_decision (multiclase), no should_failover/failover_event
│   (esas columnas no existen — pertenecían al paradigma dual-provider)
└─ Reporta distribución de las 4 clases, no solo un conteo binario
"""
import psycopg2
import pandas as pd
import numpy as np
import logging
from xgboost_optimizer import ScoringWeightOptimizer, DECISION_CLASSES

logger = logging.getLogger(__name__)


def load_training_data_from_ml_features(
    timescaledb_password,
    days=None,
    timescaledb_host='timescaledb',
    timescaledb_port=5432,
    timescaledb_db='bgp_failover_db',
    timescaledb_user='bgp_app'
):
    """
    Carga datos de entrenamiento desde ml_features (schema real, post v2).

    days=None (default) -> sin filtro de tiempo, trae TODA la tabla. Antes
    el default era 30, hardcodeado — con datasets sintéticos que ya superan
    varias semanas (ej. 7+ semanas con 100,000 ciclos), ese límite dejaba
    afuera silenciosamente la mayoría de los datos disponibles. Pasar un
    entero explícito si se quiere acotar a los últimos N días.
    """
    if days is not None:
        logger.info(f"📥 Cargando hasta {days} días de datos de ml_features...")
    else:
        logger.info(f"📥 Cargando TODA la tabla ml_features (sin límite de fecha)...")

    conn = psycopg2.connect(
        host=timescaledb_host, port=timescaledb_port, database=timescaledb_db,
        user=timescaledb_user, password=timescaledb_password
    )

    # ✅ Columnas reales de ml_features (ver migration_v2_scoring.sql).
    # degradation_cycle y provider_changed SE INCLUYEN en la query (son útiles
    # para diagnóstico/auditoría del propio dataset) pero xgboost_optimizer.
    # prepare_features() las excluye explícitamente del entrenamiento (fuga
    # de datos — ver docstring de ese archivo).
    query = f"""
    SELECT
        time, provider, cycle_number,
        -- Crudas
        peer_latency_ms, peer_loss_pct, peer_jitter_ms,
        dns1_latency_ms, dns1_jitter_ms, dns1_loss_pct,
        dns2_latency_ms, dns2_jitter_ms, dns2_loss_pct,
        -- Normalizadas (Etapa 1)
        peer_norm, dns1_norm, dns2_norm, jitter1_norm, jitter2_norm,
        loss1_norm, loss2_norm,
        base_score_dns1, base_score_dns2,
        severity_multiplier_dns1, severity_multiplier_dns2,
        score_dns1, score_dns2, max_score, quality_status,
        -- Temporales/estadísticas (Etapa 2)
        -- ⚠️ z_score_peer/cv_peer/p95_dev_peer/z_deriv_peer/latency_*_peer
        -- (sin sufijo) son las versiones AMBIGUAS anteriores a v2.3 — siguen
        -- cargándose por si hay filas históricas viejas, pero
        -- xgboost_optimizer.prepare_features() las excluye del entrenamiento
        -- (ver AMBIGUOUS_FEATURES). Las que sí se usan son las *_active/
        -- *_standby de más abajo.
        z_score_peer, z_score_dns1, z_score_dns2,
        z_score_jitter1, z_score_jitter2, z_score_loss1, z_score_loss2,
        cv_peer, cv_dns1, cv_dns2,
        p95_dev_peer, p95_dev_dns1, p95_dev_dns2,
        z_deriv_peer, z_deriv_dns1, z_deriv_dns2,
        latency_trend_5min_peer, latency_trend_5min_dns1, latency_trend_5min_dns2,
        latency_trend_15min_peer, latency_trend_15min_dns1, latency_trend_15min_dns2,
        latency_velocity_peer, latency_velocity_dns1, latency_velocity_dns2,
        latency_acceleration_peer, latency_acceleration_dns1, latency_acceleration_dns2,
        loss_spike_dns1, loss_spike_dns2,
        -- Contextuales
        hour_of_day, day_of_week, is_business_hours, is_peak_traffic, is_weekend,
        -- Contexto de cambios de provider (ya sin fuga temporal, v2.1)
        provider_changes_last_hour, time_since_last_change_min,
        -- Estado del motor (se cargan para diagnóstico; se excluyen del
        -- entrenamiento dentro de prepare_features() por fuga de datos)
        degradation_cycle, provider_changed,
        -- Target
        target_decision
    FROM ml_features
    WHERE target_decision IS NOT NULL
    {f"AND time >= NOW() - INTERVAL '{days} days'" if days is not None else ""}
    ORDER BY time
    """

    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        logger.error("❌ No se cargaron registros — ¿ml_features está vacía o el rango de días es muy chico?")
        return df

    df['time'] = pd.to_datetime(df['time'], utc=True)

    logger.info(f"✅ Cargados {len(df)} registros de ml_features")
    logger.info(f"   Fechas: {df['time'].min()} a {df['time'].max()}")
    logger.info(f"   Providers: {df['provider'].unique().tolist()}")
    logger.info(f"   Features cargadas: {len(df.columns)}")

    # Distribución real del target multiclase
    logger.info(f"\n   Distribución de target_decision:")
    counts = df['target_decision'].value_counts()
    for cls in DECISION_CLASSES:
        n = int(counts.get(cls, 0))
        pct = n / len(df) * 100 if len(df) else 0
        logger.info(f"     {cls:<15}: {n:4d} ({pct:5.1f}%)")

    unexpected = set(df['target_decision'].unique()) - set(DECISION_CLASSES)
    if unexpected:
        logger.warning(f"   ⚠️ Valores de target_decision inesperados (no en {DECISION_CLASSES}): {unexpected}")

    # Alerta de dataset chico (ver conversación: 15-20 filas/corrida en Containerlab)
    min_class_count = counts.reindex(DECISION_CLASSES).fillna(0).min()
    if min_class_count < 5:
        logger.warning(
            f"\n   ⚠️ La clase minoritaria tiene solo {int(min_class_count)} muestras. "
            f"Con tan pocos datos, cualquier métrica de CV (incluyendo esta corrida) "
            f"tiene varianza alta y debe interpretarse como orientativa, no concluyente. "
            f"Se recomienda capturar más días de datos (o generar escenarios sintéticos) "
            f"antes de usar los pesos resultantes en producción."
        )

    return df


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Entrena XGBoost multiclase desde ml_features')
    parser.add_argument('--days', type=int, default=None,
                         help='Límite de días hacia atrás (default: None = toda la tabla, sin límite)')
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 ENTRENAMIENTO XGBoost MULTICLASE — Opción A (ml_features real)")
    print("=" * 80)
    print()

    print("PASO 1: Cargar datos de ml_features")
    print("-" * 80)
    df = load_training_data_from_ml_features(
        timescaledb_password='bgp_app_password',
        days=args.days
    )
    if df.empty:
        print("❌ Sin datos — abortando.")
        return

    print(f"\n✅ Dataset cargado: {len(df)} registros | {len(df.columns)} columnas")

    print("\nPASO 2: Tuning de hiperparámetros (Bayesian Optimization)")
    print("-" * 80)
    optimizer = ScoringWeightOptimizer()
    # n_trials conservador dado el tamaño actual del dataset (ver warning arriba)
    best_params = optimizer.tune_hyperparameters(df, n_trials=15)

    print("\nPASO 3: Entrenar XGBoost con Cross-Validation")
    print("-" * 80)
    optimizer.train_with_cv(df, n_splits=5, best_params=best_params)

    print("\nPASO 4: Extraer pesos e importancia de features")
    print("-" * 80)
    weights = optimizer.get_optimized_weights()

    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print()
    print("Pesos candidatos (Gain XGBoost, renormalizados peer+dns+jitter=1):")
    for k, v in weights['candidate_weights'].items():
        actual = weights['current_weights'][k]
        print(f"  {k:8s}: {v:.4f}   (actual: {actual})")

    print("\nImportancia por familia de features:")
    for group, total in sorted(weights['group_importances'].items(), key=lambda x: -x[1]):
        print(f"  {group:20s}: {total*100:5.1f}%")

    print("\nCross-Validation Metrics (macro-average):")
    for metric, values in weights['cv_scores'].items():
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            print(f"  {metric:18s}: {np.mean(valid):.4f} ± {np.std(valid):.4f}")

    print("\nRecomendaciones:")
    for key, value in weights['recommendations'].items():
        status = "✅ SÍ" if value else "❌ NO"
        print(f"  {key}: {status}")

    print("\n⚠️ Recordatorio: estos pesos son un RANKING de importancia (Gain), no")
    print("   coeficientes lineales listos para reemplazar 0.27/0.50/0.23 en la")
    print("   fórmula. El siguiente paso (Logistic Regression) traduce esto a")
    print("   coeficientes directamente comparables — ver slide correspondiente.")


if __name__ == '__main__':
    main()
