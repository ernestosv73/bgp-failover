#!/usr/bin/env python3
"""
train_anomaly_detection.py — Detección de anomalías en DNS1/DNS2, dos etapas
═══════════════════════════════════════════════════════════════════════════════
Pipeline INDEPENDIENTE del motor de failover — lee de link_health_features /
link_health_ground_truth (pobladas por link_health_monitor.py +
link_health_feature_engine.py), no de ml_features/bgp_metrics_new.

MOTIVACIÓN (ver conversación — paper "Detecting Anomalies in Network Latency
Time Series: From Statistical Filters to Machine Learning"):

El target (is_anomaly / episode_type) es VERDAD DE TERRENO del generador
sintético (sabemos exactamente qué ciclos inyectamos como anómalos, y de
qué tipo) — no una fórmula de scoring. Esto evita la circularidad de usar
como feature algo derivado de la misma regla que genera la etiqueta.

✅ DISEÑO:
├─ Formato LARGO: una fila por (ciclo, dns_target) — dns1 y dns2 son
│   observaciones intercambiables del mismo problema univariado.
├─ Etapa 1 (binaria): P(is_anomaly | features) — XGBoost + Logistic Regression.
├─ Etapa 2 (multiclase, SOLO sobre filas con is_anomaly=1):
│   P(episode_type | is_anomaly=1, features) — 5 clases:
│   step / spike / unstable / oscillating / slow_increase.
├─ Combinado: P(tipo) = P(is_anomaly) × P(tipo | is_anomaly) — ver fórmula
│   del paper.
└─ ⚠️ NO incluye:
   - 'peer' — no se monitorea en este pipeline (el hop queda implícito en
     la medición end-to-end a los DNS).
   - Features normalizadas contra umbral fijo (*_norm) — el objetivo es
     ESTIMAR ese umbral, no asumirlo (ver migration_link_health.sql).
   - 'provider' — un solo enlace, no hay dual-provider.

USO:
    python3 train_anomaly_detection.py
    python3 train_anomaly_detection.py --days 60

═══════════════════════════════════════════════════════════════════════════════
📚 REFERENCIA — Umbrales iniciales (NO usados como feature en este pipeline)
═══════════════════════════════════════════════════════════════════════════════
Estos son los umbrales con los que arrancó todo el proyecto (fórmula de
scoring del motor de failover, definidos a partir del draft IETF
"BGP Performance-aware Routing Mechanism"):

    DNS1: warning=15.0ms, critical=30.0ms
    DNS2: warning=30.0ms, critical=60.0ms

Este pipeline NO los usa para normalizar features (ver migration_link_health.sql
— por diseño, sin *_norm) porque el objetivo es dejar que z_score/cv/p95_dev
le indiquen al modelo qué es anómalo de forma relativa a la ventana móvil
reciente, sin asumir de antemano dónde está el límite en milisegundos. Se
documentan acá como punto de referencia para interpretar los resultados: si
el modelo aprende a marcar anomalías sistemáticamente cerca de estos
valores, es una validación cruzada de que el umbral original estaba bien
calibrado. Si aprende algo sistemáticamente distinto, es una señal de que
convendría revisarlo — exactamente el tipo de conclusión que este pipeline
está diseñado para poder sacar.

═══════════════════════════════════════════════════════════════════════════════
📚 REFERENCIA — Taxonomía de anomalías (paper "Detecting Anomalies in
Network Latency Time Series: From Statistical Filters to Machine Learning")
═══════════════════════════════════════════════════════════════════════════════
episode_type mapea directamente a los "anomaly waveforms" del paper (más
'oscillating', que ellos no nombran como waveform explícito, aunque sí
mencionan "periodic" como uno de los tipos de serie BASE):

    step           <- "sudden increase to a higher value, remaining at that
                       level for [tiempo], before returning to baseline"
    spike          <- "sudden increase... for a couple of timestamps before
                       returning to baseline"
    slow_increase  <- "latency slowly drifts to a higher value over a long
                       period of time, rather than a sudden jump"
    unstable       <- "large standard dev in comparison to the mean [CV
                       alto], before returning to baseline"
    oscillating    <- alternancia periódica alto/bajo (no está en la lista
                       de waveforms del paper, pero "periodic" sí aparece
                       como tipo de serie base — ver Figura 4)

DEFAULT_EPISODE_TYPE_WEIGHTS (ver ScenarioGenerator en
synthetic_data_generator.py) es la probabilidad de que, al arrancar un
episodio nuevo, se elija cada tipo — NO es directamente "qué proporción del
dataset final va a tener cada tipo": tipos con rampas/mesetas más largas
(step, slow_increase) acumulan más FILAS por episodio que tipos breves
(spike), aunque arranquen con probabilidades similares.

═══════════════════════════════════════════════════════════════════════════════
📚 REFERENCIA — Modelo de ruido de fondo (Ornstein-Uhlenbeck)
═══════════════════════════════════════════════════════════════════════════════
El paper simula series base con kernels de Gaussian Process (mencionado de
paso, sin profundizar en la implementación). Acá se usa Ornstein-Uhlenbeck
— la aproximación discretizada estándar de un GP con kernel exponencial, a
costo O(1) por muestra (ver clase OUProcess en synthetic_data_generator.py).

Por qué importa para ESTE pipeline en particular: el tráfico "normal"
(clase mayoritaria, la que el modelo tiene que aprender a diferenciar de
una anomalía real) necesita tener memoria temporal — la latencia real de
ahora está correlacionada con la de hace un minuto, no es ruido
independiente ciclo a ciclo. Con ruido blanco (i.i.d., el comportamiento
anterior a esta versión), las features de tendencia/velocidad
(trend_5min, velocity) tendrían falsos positivos frecuentes por pura
casualidad estadística en los ciclos normales. Con OU, el "normal" tiene
una textura temporal más realista, así que cuando el modelo aprende a
distinguirlo de una degradación genuina, la distinción es más significativa.

theta (velocidad de reversión a la media, en 1/ciclos) controla la memoria:
memoria_en_ciclos ≈ 1/theta. El default (theta=1/3 ≈ 0.333) da una memoria
de ~3 ciclos (~2.25 min a 45s/ciclo) — configurable vía --ou-theta en
link_health_monitor.py.
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import logging

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EPISODE_TYPES = ['step', 'spike', 'unstable', 'oscillating', 'slow_increase']
SUSTAINED_TYPES = {'step', 'slow_increase'}

# ✅ Umbrales INICIALES de referencia (fórmula de scoring original del motor
# de failover, ver bgp_failover_engine_new.py DNS_THRESHOLDS) — NO se usan
# como feature en este pipeline (ver docstring del módulo), solo sirven
# como punto de comparación al interpretar los resultados.
REFERENCE_THRESHOLDS_MS = {
    'dns1': {'warning': 15.0, 'critical': 30.0},
    'dns2': {'warning': 30.0, 'critical': 60.0},
}

BOOLEAN_FEATURES = ['loss_spike']

# ✅ v2 — esquema FCC (paper "Where Has the Time Gone?..."): reemplaza
# is_business_hours/is_peak_traffic/is_weekend por dos categóricas.
CATEGORICAL_CONTEXT_FEATURES = {
    'time_category': ['peak', 'off_peak', 'satsun'],
    'daypart': ['00-06', '06-12', '12-18', '18-20', '20-22', '22-24', '00-12', '12-24'],
}

GENERIC_DNS_FEATURES = [
    'latency_ms', 'jitter_ms', 'loss_pct',
    'z_score', 'z_score_jitter', 'z_score_loss', 'cv', 'p95_dev', 'z_deriv',
    'trend_5min', 'trend_15min', 'velocity', 'acceleration', 'loss_spike',
]
# ⚠️ hour_of_day/day_of_week NO se usan como feature del modelo — quedan
# solo en la tabla (link_health_features) para diagnóstico/crosstabs. Se
# excluyen acá porque, al tener resolución completa (24/7 valores), le
# daban al modelo un segundo camino para explotar la misma ventana horaria
# fija del generador, además de time_category/daypart — concentrando en vez
# de reducir la dependencia temporal (ver conversación: hour_of_day llegó a
# ser el 43.2% de la importancia en XGBoost). Con esto, el modelo entrena
# ÚNICAMENTE con la resolución que da el esquema FCC, no con la hora exacta.
SHARED_CONTEXT_FEATURES = ['time_category', 'daypart']


def _rename_map_for_dns(dns_name, num):
    return {
        f'{dns_name}_latency_ms': 'latency_ms',
        f'{dns_name}_jitter_ms': 'jitter_ms',
        f'{dns_name}_loss_pct': 'loss_pct',
        f'z_score_{dns_name}': 'z_score',
        f'z_score_jitter{num}': 'z_score_jitter',
        f'z_score_loss{num}': 'z_score_loss',
        f'cv_{dns_name}': 'cv',
        f'p95_dev_{dns_name}': 'p95_dev',
        f'z_deriv_{dns_name}': 'z_deriv',
        f'latency_trend_5min_{dns_name}': 'trend_5min',
        f'latency_trend_15min_{dns_name}': 'trend_15min',
        f'latency_velocity_{dns_name}': 'velocity',
        f'latency_acceleration_{dns_name}': 'acceleration',
        f'loss_spike_{dns_name}': 'loss_spike',
    }


def load_long_format_dataset(timescaledb_password, days=None,
                              timescaledb_host='timescaledb', timescaledb_port=5432,
                              timescaledb_db='bgp_failover_db', timescaledb_user='bgp_app'):
    """
    Carga link_health_features + link_health_ground_truth y arma el dataset
    en formato LARGO (una fila por ciclo×dns_target). days=None -> toda la tabla.
    """
    logger.info("📥 Cargando link_health_features + link_health_ground_truth...")
    conn = psycopg2.connect(
        host=timescaledb_host, port=timescaledb_port, database=timescaledb_db,
        user=timescaledb_user, password=timescaledb_password
    )

    time_filter = f"WHERE time >= NOW() - INTERVAL '{days} days'" if days is not None else ""

    wide_cols = (
        list(_rename_map_for_dns('dns1', '1').keys()) +
        list(_rename_map_for_dns('dns2', '2').keys()) +
        SHARED_CONTEXT_FEATURES + ['time', 'cycle_number']
    )
    wide_query = f"""
        SELECT {', '.join(sorted(set(wide_cols)))}
        FROM link_health_features
        {time_filter}
        ORDER BY time
    """
    try:
        wide = pd.read_sql(wide_query, conn)
    except Exception as e:
        conn.close()
        raise RuntimeError(
            f"Error cargando link_health_features: {e}\n"
            f"¿Corriste migration_link_health.sql y link_health_feature_engine.py?"
        )

    gt_query = f"""
        SELECT cycle_number, dns_target, episode_type, is_anomaly, is_sustained
        FROM link_health_ground_truth
        ORDER BY cycle_number
    """
    try:
        gt = pd.read_sql(gt_query, conn)
    except Exception as e:
        conn.close()
        raise RuntimeError(
            f"Error cargando link_health_ground_truth: {e}\n"
            f"¿Corriste link_health_monitor.py --mode synthetic? (--mode real no puebla "
            f"esta tabla — ver docstring de ese script)."
        )
    conn.close()

    if wide.empty or gt.empty:
        logger.warning("⚠️ ml_features o anomaly_ground_truth están vacías")
        return pd.DataFrame()

    logger.info(f"✅ link_health_features: {len(wide)} ciclos | link_health_ground_truth: {len(gt)} filas")

    views = []
    for dns_name, num in [('dns1', '1'), ('dns2', '2')]:
        rename_map = _rename_map_for_dns(dns_name, num)
        cols_needed = list(rename_map.keys()) + SHARED_CONTEXT_FEATURES + ['time', 'cycle_number']
        sub = wide[cols_needed].rename(columns=rename_map).copy()
        sub['dns_target'] = dns_name
        views.append(sub)

    long_df = pd.concat(views, ignore_index=True)
    df = long_df.merge(gt, on=['cycle_number', 'dns_target'], how='inner')
    df['time'] = pd.to_datetime(df['time'], utc=True)

    logger.info(f"✅ Dataset largo construido: {len(df)} filas (2× ciclos, una por dns_target)")
    return df


def prepare_features(df):
    """Prepara X (features genéricas + contexto) e y (is_anomaly). Pipeline
    de un solo enlace: no hay 'provider' que codificar (ver docstring del módulo).

    time_category/daypart se codifican one-hot con categorías FIJAS (no
    inferidas del subset recibido) — así las columnas de X son consistentes
    entre folds/subconjuntos aunque alguno no tenga ejemplos de alguna
    categoría (ej. la Etapa 2, con pocas filas, podría no tener ninguna fila
    en 'satsun' en un fold dado)."""
    numeric_context = [c for c in SHARED_CONTEXT_FEATURES if c not in CATEGORICAL_CONTEXT_FEATURES]
    candidate_cols = GENERIC_DNS_FEATURES + numeric_context
    X = df[candidate_cols].copy()

    for col in BOOLEAN_FEATURES:
        if col in X.columns and X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    for col, categories in CATEGORICAL_CONTEXT_FEATURES.items():
        if col in df.columns:
            cat_series = pd.Categorical(df[col], categories=categories)
            dummies = pd.get_dummies(cat_series, prefix=col)
            dummies.index = X.index
            X = pd.concat([X, dummies], axis=1)

    X = X.fillna(0)
    object_cols = X.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    bool_dummy_cols = X.select_dtypes(include='bool').columns.tolist()
    for col in bool_dummy_cols:
        X[col] = X[col].astype(int)

    y_is_anomaly = df['is_anomaly'].astype(int).values
    return X, y_is_anomaly, list(X.columns)


def _cv_binary_or_multiclass(X, y, n_splits, model_kind, num_class=None, random_state=42):
    """
    Cross-validation genérico (XGBoost o Logistic Regression), binario o
    multiclase según num_class. Devuelve métricas macro + importancias/
    coeficientes promediados entre folds.
    """
    class_counts = np.bincount(y)
    min_class_count = class_counts[class_counts > 0].min()
    if min_class_count < n_splits:
        n_splits = max(2, min_class_count)
        logger.warning(f"⚠️ Clase minoritaria chica -> n_splits ajustado a {n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = {'precision_macro': [], 'recall_macro': [], 'f1_macro': [], 'roc_auc': []}
    importances = []

    is_multiclass = num_class is not None and num_class > 2

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sample_weight = compute_sample_weight('balanced', y_train)

        if model_kind == 'xgboost':
            params = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbosity=0)
            if is_multiclass:
                model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_class, **params)
            else:
                model = xgb.XGBClassifier(objective='binary:logistic', **params)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            importances.append(model.feature_importances_)

        elif model_kind == 'logreg':
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)
            importances.append(np.abs(model.coef_).mean(axis=0) if is_multiclass else np.abs(model.coef_[0]))

        metrics['precision_macro'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        try:
            if is_multiclass:
                present = np.unique(y_test)
                metrics['roc_auc'].append(
                    roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro',
                                   labels=list(range(num_class))) if len(present) >= 2 else np.nan
                )
            else:
                metrics['roc_auc'].append(roc_auc_score(y_test, y_proba[:, 1]))
        except Exception:
            metrics['roc_auc'].append(np.nan)

        logger.info(f"   Fold {fold}/{n_splits}: Precision={metrics['precision_macro'][-1]:.3f} | "
                    f"Recall={metrics['recall_macro'][-1]:.3f} | F1={metrics['f1_macro'][-1]:.3f} | "
                    f"ROC-AUC={metrics['roc_auc'][-1]:.3f}")

    mean_importance = np.mean(np.stack(importances), axis=0)
    return metrics, mean_importance


def print_top_features(feature_names, importances, top_n=15):
    order = np.argsort(importances)[::-1][:top_n]
    total = importances.sum() or 1.0
    for rank, idx in enumerate(order, 1):
        pct = importances[idx] / total * 100
        bar = "█" * int(pct / 2)
        logger.info(f"   {rank:2d}. {feature_names[idx]:20s} {bar:20s} {pct:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Detección de anomalías en 2 etapas (independiente del motor de failover)')
    parser.add_argument('--days', type=int, default=None, help='Límite de días hacia atrás (default: toda la tabla)')
    parser.add_argument('--n-splits', type=int, default=5, help='Folds de Cross-Validation (default: 5)')
    args = parser.parse_args()

    print("=" * 80)
    print("🔍 DETECCIÓN DE ANOMALÍAS EN DOS ETAPAS — independiente de target_decision")
    print("=" * 80)
    print("\n📚 Umbrales iniciales de referencia (NO usados como feature — ver docstring):")
    print(f"   DNS1: warning={REFERENCE_THRESHOLDS_MS['dns1']['warning']}ms  "
          f"critical={REFERENCE_THRESHOLDS_MS['dns1']['critical']}ms")
    print(f"   DNS2: warning={REFERENCE_THRESHOLDS_MS['dns2']['warning']}ms  "
          f"critical={REFERENCE_THRESHOLDS_MS['dns2']['critical']}ms")
    print("   (fórmula de scoring original del motor de failover — draft IETF)")

    print("\nPASO 1: Cargar y reformular el dataset (formato largo)")
    print("-" * 80)
    df = load_long_format_dataset(timescaledb_password='bgp_app_password', days=args.days)
    if df.empty:
        print("❌ Sin datos — abortando.")
        return

    print(f"\n✅ Dataset: {len(df)} filas (ciclo × dns_target)")
    print(f"   is_anomaly=True: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
    print(f"   is_sustained=True: {df['is_sustained'].sum()}")
    print("\n   Distribución de episode_type (solo filas anómalas):")
    print(df.loc[df['is_anomaly'], 'episode_type'].value_counts().to_string())

    # ✅ Crosstab de diagnóstico — esquema FCC (time_category/daypart).
    # Muestra si hay concentración de anomalías en franjas específicas, con
    # la resolución asimétrica del paper (más fina cerca del horario pico).
    print("\n   Anomalías por time_category (peak/off_peak/satsun):")
    print(pd.crosstab(df['time_category'], df['is_anomaly']).to_string())

    print("\n   Anomalías por daypart:")
    daypart_order = CATEGORICAL_CONTEXT_FEATURES['daypart']
    ct_daypart = pd.crosstab(df['daypart'], df['is_anomaly'])
    ct_daypart = ct_daypart.reindex([d for d in daypart_order if d in ct_daypart.index])
    print(ct_daypart.to_string())

    print("\n   episode_type por daypart (solo filas anómalas):")
    ct_type = pd.crosstab(
        df.loc[df['is_anomaly'], 'daypart'],
        df.loc[df['is_anomaly'], 'episode_type']
    )
    ct_type = ct_type.reindex([d for d in daypart_order if d in ct_type.index])
    print(ct_type.to_string())

    # ════════════════════════════════════════════════════════════════════
    # ETAPA 1 — binaria: P(is_anomaly | features)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PASO 2: Etapa 1 — P(is_anomaly | features)")
    print("=" * 80)
    X, y_anomaly, feature_names = prepare_features(df)
    print(f"Features usadas ({len(feature_names)}): {feature_names}")

    print("\n🤖 XGBoost (Etapa 1)")
    print("-" * 80)
    xgb_metrics_1, xgb_importance_1 = _cv_binary_or_multiclass(X, y_anomaly, args.n_splits, 'xgboost')
    print("\nTop features (Etapa 1, XGBoost):")
    print_top_features(feature_names, xgb_importance_1)

    print("\n📐 Logistic Regression (Etapa 1)")
    print("-" * 80)
    lr_metrics_1, lr_importance_1 = _cv_binary_or_multiclass(X, y_anomaly, args.n_splits, 'logreg')
    print("\nTop |coeficientes| (Etapa 1, Logistic Regression):")
    print_top_features(feature_names, lr_importance_1)

    # ════════════════════════════════════════════════════════════════════
    # ETAPA 2 — multiclase: P(episode_type | is_anomaly=1, features)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PASO 3: Etapa 2 — P(episode_type | is_anomaly=1, features)")
    print("=" * 80)
    df_anomalous = df[df['is_anomaly']].copy()
    class_counts = df_anomalous['episode_type'].value_counts()
    print(f"Filas anómalas disponibles: {len(df_anomalous)}")
    print(class_counts.to_string())

    valid_types = class_counts[class_counts >= args.n_splits].index.tolist()
    if len(valid_types) < 2:
        print("\n❌ No hay suficientes tipos de episodio con muestras mínimas para la Etapa 2 — abortando esa etapa.")
        xgb_metrics_2 = lr_metrics_2 = None
        xgb_importance_2 = lr_importance_2 = None
        type_to_idx = {}
    else:
        df_anomalous = df_anomalous[df_anomalous['episode_type'].isin(valid_types)]
        type_to_idx = {t: i for i, t in enumerate(sorted(valid_types))}
        X2, _, _ = prepare_features(df_anomalous)
        y_type = df_anomalous['episode_type'].map(type_to_idx).values
        num_class = len(type_to_idx)

        print(f"\n🤖 XGBoost (Etapa 2, {num_class} clases: {sorted(valid_types)})")
        print("-" * 80)
        xgb_metrics_2, xgb_importance_2 = _cv_binary_or_multiclass(
            X2, y_type, min(args.n_splits, class_counts[valid_types].min()), 'xgboost', num_class=num_class)
        print("\nTop features (Etapa 2, XGBoost):")
        print_top_features(feature_names, xgb_importance_2)

        print(f"\n📐 Logistic Regression (Etapa 2)")
        print("-" * 80)
        lr_metrics_2, lr_importance_2 = _cv_binary_or_multiclass(
            X2, y_type, min(args.n_splits, class_counts[valid_types].min()), 'logreg', num_class=num_class)
        print("\nTop |coeficientes| (Etapa 2, Logistic Regression):")
        print_top_features(feature_names, lr_importance_2)

    # ════════════════════════════════════════════════════════════════════
    # RESUMEN
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("✅ RESUMEN")
    print("=" * 80)

    def _fmt(metrics):
        return {k: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}" for k, v in metrics.items()}

    print("\nEtapa 1 (is_anomaly) — XGBoost:", _fmt(xgb_metrics_1))
    print("Etapa 1 (is_anomaly) — Logistic Regression:", _fmt(lr_metrics_1))
    if xgb_metrics_2:
        print("\nEtapa 2 (episode_type) — XGBoost:", _fmt(xgb_metrics_2))
        print("Etapa 2 (episode_type) — Logistic Regression:", _fmt(lr_metrics_2))

    # ✅ Comparación empírica contra los umbrales iniciales de referencia:
    # ¿la latencia observada durante anomalías SOSTENIDAS (step/slow_increase
    # — las que el proyecto considera degradación real) se ubica cerca del
    # umbral crítico original, o sistemáticamente distinta? No reemplaza un
    # análisis estadístico riguroso, pero da una primera pista rápida.
    print("\n📊 Latencia observada durante anomalías SOSTENIDAS vs. umbral de referencia:")
    for dns_name in ('dns1', 'dns2'):
        sustained = df[(df['dns_target'] == dns_name) & (df['is_sustained'])]
        if len(sustained) > 0:
            mean_lat = sustained['latency_ms'].mean()
            median_lat = sustained['latency_ms'].median()
            ref_critical = REFERENCE_THRESHOLDS_MS[dns_name]['critical']
            print(f"   {dns_name}: media={mean_lat:.1f}ms | mediana={median_lat:.1f}ms | "
                  f"umbral crítico de referencia={ref_critical}ms | "
                  f"n={len(sustained)}")
        else:
            print(f"   {dns_name}: sin filas sostenidas suficientes en este dataset")

    print("\n⚠️ Lectura: este pipeline usa un target de VERDAD DE TERRENO (no la")
    print("   fórmula del motor), no monitorea peer, y no asume ningún umbral fijo —")
    print("   los resultados NO son directamente comparables a los de")
    print("   train_logistic_regression.py (son dos preguntas distintas, sobre")
    print("   pipelines de datos completamente independientes).")
    print("=" * 80)


if __name__ == '__main__':
    main()
