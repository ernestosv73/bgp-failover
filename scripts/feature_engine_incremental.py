#!/usr/bin/env python3
"""
Feature Engine v2 — Etapa 2: features temporales por métrica desde bgp_metrics_new
✅ REESCRITO COMPLETO (v1 leía de la tabla legacy `bgp_metrics`, con DNS singular
   y should_failover/failover_event basados en monitoreo dual-provider — ambos
   incompatibles con la arquitectura actual del motor v2.4).

LÓGICA:
├─ Lee de bgp_metrics_new (NO de bgp_metrics). Un registro por ciclo, un solo
│   provider activo monitoreado — no hay "score del provider alternativo".
├─ NO recalcula score/quality_index — usa los campos que el motor v2.4 ya
│   computó correctamente (peer_norm, dns1_norm, ..., max_score, quality_status).
├─ z-score, CV (σ_ratio), desviación de p95 y derivada del Z se calculan
│   POR MÉTRICA (peer, dns1, dns2, jitter1, jitter2, loss1, loss2) — un valor
│   agregado diluiría la señal de una sola métrica degradándose, igual que
│   diluía el score ponderado antes del fix de "breach individual" (v2.2).
├─ Tendencia/velocidad/aceleración se calculan para peer/dns1/dns2 (las tres
│   señales de latencia — las que en la práctica dispararon nuestros
│   failovers reales).
├─ Ventana de contexto histórico para rolling stats: TREND_WINDOW_LONG=30
│   ciclos — distinta de la ventana de 3 ciclos del
│   motor en vivo (esa está atada a la confirmación de failover/retorno,
│   no a "testing rápido"; ver documentación en bgp_failover_engine_new.py).
├─ Target: `target_decision` (multiclase: normal/degradacion/failover/retorno),
│   derivado directamente de bgp_metrics_new.decision. Filas con
│   decision IN ('error', 'failover_inmediato') se EXCLUYEN del insert —
│   no participan del entrenamiento.
└─ Modo incremental con ventana de contexto: para que las rolling stats de las
   primeras filas nuevas de cada corrida no arranquen "en frío", se traen
   también las TREND_WINDOW_LONG-1 filas anteriores al último timestamp
   procesado, se usan solo como contexto de cálculo, y NO se re-insertan.

✅ v2.2 — cold-start real (ml_features vacía): en vez de limitarse a
"últimas LAST_HOURS horas" (que dejaba afuera casi todo un dataset generado
de una sola vez, ej. el generador sintético en escala 'realistic' con
semanas/meses de datos), ahora lee bgp_metrics_new COMPLETA. No necesita
ventana de contexto adicional en este modo — el dataset ya arranca desde el
principio.
"""
import psycopg2
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

# === Configuración ===
TIMESCALEDB_HOST = 'timescaledb'
TIMESCALEDB_PORT = 5432
TIMESCALEDB_DB = 'bgp_failover_db'
TIMESCALEDB_USER = 'bgp_app'
TIMESCALEDB_PASSWORD = 'bgp_app_password'

# === Configuración de Feature Engine ===
EXECUTION_MODE = "incremental"
# ✅ v2.2: LAST_HOURS eliminado — el cold-start (ml_features vacía) ahora lee
# la tabla bgp_metrics_new COMPLETA en vez de una ventana de horas fija (ver
# load_metrics_incremental). Necesario para datasets sintéticos generados de
# una sola vez que pueden abarcar semanas/meses.

# ✅ Ventanas de tendencia (Etapa 2) — distintas de la ventana de 3 ciclos del
# motor en vivo (esa es de CONFIRMACIÓN de decisión; esta es de TENDENCIA para
# features de entrenamiento).
#
# ⚠️ Corrección: el intervalo REAL entre ciclos NO es CYCLE_INTERVAL (30s) —
# ese valor es solo el sleep() entre mediciones; cada ciclo también tarda
# ~15s en ejecutar las dos corridas de MTR (DNS1+DNS2). Medido empíricamente
# sobre una corrida real: intervalo real = 44.9s ± 0.06s, no 30s.
# Por eso estas ventanas se describen en CANTIDAD DE CICLOS, no en minutos —
# una estimación en minutos asumiendo 30s/ciclo subestimaría el horizonte
# real en ~50%. El cálculo en sí (rolling sobre N filas) es correcto
# independientemente de esto; es solo la etiqueta en minutos la que era
# engañosa.
CYCLE_INTERVAL_SECONDS = 30       # sleep() configurado en el motor (referencia, NO el gap real)
TREND_WINDOW_SHORT = 10   # ~10 ciclos (empíricamente ~7.5 min, no 5 min)
TREND_WINDOW_LONG = 30    # ~30 ciclos (empíricamente ~22.5 min, no 15 min)

# ✅ Umbral de spike de pérdida (diferencia entre ciclos consecutivos de la
# pérdida ya agregada en ventana, dns1_loss_pct/dns2_loss_pct de bgp_metrics_new)
LOSS_SPIKE_THRESHOLD_PCT = 5.0

# ✅ Decisiones que NO participan del entrenamiento (ver conversación):
#    'error'             -> falla de medición, no es una observación válida
#    'failover_inmediato' -> bypass de seguridad (pérdida severa en 1 ciclo),
#                            mecanismo de disparo distinto al de degradación
#                            sostenida; mezclarlo confundiría al modelo sobre
#                            qué patrón temporal antecede a un failover típico.
EXCLUDED_DECISIONS = ('error', 'failover_inmediato')

# Métricas de latencia con features de tendencia/velocidad/aceleración
LATENCY_METRICS = [
    ('peer', 'peer_latency_ms'),
    ('dns1', 'dns1_latency_ms'),
    ('dns2', 'dns2_latency_ms'),
]

# Métricas con z-score (todas las señales normalizables — por métrica, no agregado)
ZSCORE_METRICS = [
    ('peer', 'peer_latency_ms'),
    ('dns1', 'dns1_latency_ms'),
    ('dns2', 'dns2_latency_ms'),
    ('jitter1', 'dns1_jitter_ms'),
    ('jitter2', 'dns2_jitter_ms'),
    ('loss1', 'dns1_loss_pct'),
    ('loss2', 'dns2_loss_pct'),
]

# Métricas con CV (σ_ratio) y desviación de p95 — las tres de latencia,
# que son las que en la práctica dispararon nuestros failovers reales
CV_P95_METRICS = [
    ('peer', 'peer_latency_ms'),
    ('dns1', 'dns1_latency_ms'),
    ('dns2', 'dns2_latency_ms'),
]


class TimescaleDBClient:
    """Cliente para TimescaleDB con soporte de lectura incremental con contexto."""

    def __init__(self, host, port, database, user, password):
        self.conn = psycopg2.connect(
            host=host, port=port, database=database, user=user, password=password
        )
        logging.info(f"✅ Conectado a TimescaleDB en {host}:{port}")

    def get_last_feature_timestamp(self):
        """Lee el último timestamp ya procesado en ml_features."""
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT COALESCE(MAX(time), NULL) FROM ml_features")
            result = cur.fetchone()
            cur.close()
            if result[0]:
                logging.info(f"✅ Último timestamp en ml_features: {result[0]}")
                return result[0]
            logging.info("ℹ️ ml_features vacía, se procesará la tabla bgp_metrics_new completa")
            return None
        except Exception as e:
            logging.error(f"⚠️ Error leyendo last_timestamp: {e}")
            return None

    def insert_ml_features(self, row: pd.Series) -> bool:
        """Inserta un registro de features en ml_features."""
        try:
            cur = self.conn.cursor()
            clean = row.where(pd.notnull(row), None)
            columns = list(clean.index)
            placeholders = ", ".join(["%s"] * len(columns))
            column_names = ", ".join(columns)
            query = f"INSERT INTO ml_features ({column_names}) VALUES ({placeholders})"
            values = [clean[col] for col in columns]
            cur.execute(query, values)
            self.conn.commit()
            cur.close()
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error insertando feature: {e}")
            return False


class FeatureEngineV2:
    """
    Feature Engine v2 — deriva features temporales por métrica desde
    bgp_metrics_new hacia ml_features, para consumo de Etapa 3 (XGBoost).
    """

    def __init__(self):
        self.ts_client = TimescaleDBClient(
            host=TIMESCALEDB_HOST, port=TIMESCALEDB_PORT, database=TIMESCALEDB_DB,
            user=TIMESCALEDB_USER, password=TIMESCALEDB_PASSWORD
        )
        self.conn = self.ts_client.conn

    # ------------------------------------------------------------------
    # Carga incremental CON ventana de contexto (fix del cold-start rolling)
    # ------------------------------------------------------------------
    def load_metrics_incremental(self):
        """
        Carga los datos desde bgp_metrics_new. Dos modos:

        1. COLD START (ml_features vacía, last_timestamp=None): lee la tabla
           COMPLETA, sin filtro de tiempo. Necesario para datasets generados
           de una sola vez que pueden abarcar semanas/meses (ej. el generador
           sintético en modo 'realistic') — el viejo comportamiento de
           "últimas LAST_HOURS horas" dejaba afuera prácticamente todo el
           dataset en ese escenario. No hace falta ventana de contexto en
           este modo: el dataset ya arranca desde el principio, y las
           rolling stats de las primeras filas simplemente tienen menos
           historia disponible (min_periods=3 ya lo maneja sin arrancar
           en frío con NaN).

        2. INCREMENTAL (ya hay checkpoint en ml_features): comportamiento
           sin cambios — trae solo lo nuevo desde el último timestamp
           procesado, más TREND_WINDOW_LONG-1 filas previas como contexto
           para que las rolling stats de las primeras filas nuevas del
           batch no arranquen sin ventana.
        """
        last_timestamp = self.ts_client.get_last_feature_timestamp()
        is_cold_start = last_timestamp is None

        base_select = """
                time, cycle_number, current_provider AS provider,
                peer_latency_ms, peer_jitter_ms, peer_loss_pct,
                dns1_latency_ms, dns1_jitter_ms, dns1_loss_pct,
                dns2_latency_ms, dns2_jitter_ms, dns2_loss_pct,
                peer_norm, dns1_norm, dns2_norm, jitter1_norm, jitter2_norm,
                loss1_norm, loss2_norm,
                base_score_dns1, base_score_dns2,
                severity_multiplier_dns1, severity_multiplier_dns2,
                score_dns1, score_dns2, max_score,
                quality_status, degradation_cycle, provider_changed, decision
        """

        if is_cold_start:
            logging.info("📥 Primera ejecución (ml_features vacía): cargando la tabla COMPLETA bgp_metrics_new...")
            query = f"""
                SELECT {base_select}
                FROM bgp_metrics_new
                ORDER BY time
            """
        else:
            time_filter = f"'{last_timestamp}'::timestamptz"
            logging.info(f"📥 Cargando datos después de: {last_timestamp}")
            query = f"""
                SELECT {base_select}
                FROM bgp_metrics_new
                WHERE time > {time_filter}
                ORDER BY time
            """

        try:
            df_new = pd.read_sql(query, self.conn)
            # ✅ Fix: con conexión psycopg2 "cruda" (no SQLAlchemy), pd.read_sql
            # no vectoriza columnas timestamp — quedan dtype=object con
            # datetime.datetime sueltos, y .dt falla más adelante. Se fuerza
            # la conversión explícita apenas se lee.
            df_new['time'] = pd.to_datetime(df_new['time'], utc=True)
        except Exception as e:
            logging.error(f"Error cargando métricas nuevas: {e}")
            return pd.DataFrame()

        if df_new.empty:
            logging.info("ℹ️ No hay datos en bgp_metrics_new para procesar")
            return df_new

        df_new['is_context'] = False

        if is_cold_start:
            logging.info(f"✅ Cargados {len(df_new)} registros (tabla completa)")
            if len(df_new) > 200_000:
                logging.warning(
                    f"⚠️ {len(df_new)} filas es un volumen considerable para procesar en memoria "
                    f"de una sola vez. Si esto se vuelve un problema de rendimiento/memoria, "
                    f"considerar procesar por bloques de tiempo en vez de la tabla completa."
                )
            # Sin ventana de contexto: el dataset ya arranca desde el
            # principio, no hay nada "antes" que traer.
            df_context = pd.DataFrame()
        else:
            logging.info(f"✅ Cargados {len(df_new)} registros nuevos")

            # ✅ Ventana de contexto: últimas TREND_WINDOW_LONG-1 filas ANTES
            # del primer registro nuevo, para que rolling/z-score/CV no
            # arranquen en frío.
            context_needed = TREND_WINDOW_LONG - 1
            if context_needed > 0:
                first_new_time = df_new['time'].min()
                context_query = f"""
                    SELECT {base_select}
                    FROM bgp_metrics_new
                    WHERE time < '{first_new_time}'::timestamptz
                    ORDER BY time DESC
                    LIMIT {context_needed}
                """
                try:
                    df_context = pd.read_sql(context_query, self.conn)
                    df_context['time'] = pd.to_datetime(df_context['time'], utc=True)
                    df_context['is_context'] = True
                    logging.info(f"📎 Contexto histórico agregado: {len(df_context)} filas previas (no se insertan)")
                except Exception as e:
                    logging.warning(f"⚠️ No se pudo cargar contexto histórico: {e}")
                    df_context = pd.DataFrame()
            else:
                df_context = pd.DataFrame()

        df = pd.concat([df_context, df_new], ignore_index=True).sort_values('time').reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Etapa 2 — features por métrica: z-score, CV, desviación p95, derivada Z
    # ------------------------------------------------------------------
    def calculate_zscore_features(self, df):
        """
        z-score POR MÉTRICA (rolling, ventana TREND_WINDOW_LONG). Un z-score
        agregado diluiría la señal de una sola métrica degradándose — misma
        razón por la que en v2.2 el score ponderado no bastaba y hubo que
        agregar la detección de breach individual.
        """
        if df.empty:
            return df
        logging.info("🔧 Calculando z-score por métrica...")
        df = df.sort_values('time').copy()

        for name, col in ZSCORE_METRICS:
            roll_mean = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).mean()
            roll_std = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).std(ddof=0)
            z = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
            df[f'z_score_{name}'] = z

            # Derivada del Z (Etapa 2): aceleración de la degradación —
            # sube más rápido en degradación progresiva que en fluctuación normal
            if name in ('peer', 'dns1', 'dns2'):
                df[f'z_deriv_{name}'] = df[f'z_score_{name}'].diff()

        return df

    def calculate_cv_p95_features(self, df):
        """
        CV (σ_ratio, %) y desviación del p95 respecto a la media móvil,
        por métrica de latencia (peer/dns1/dns2).
        """
        if df.empty:
            return df
        logging.info("🔧 Calculando CV (σ_ratio) y desviación de p95...")
        df = df.sort_values('time').copy()

        for name, col in CV_P95_METRICS:
            roll_mean = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).mean()
            roll_std = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).std(ddof=0)
            roll_p95 = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).quantile(0.95)

            df[f'cv_{name}'] = (roll_std / roll_mean.replace(0, np.nan)) * 100
            # Desviación relativa del valor actual respecto al p95 móvil:
            # >0 significa que el valor actual ya superó el p95 histórico reciente
            df[f'p95_dev_{name}'] = (df[col] - roll_p95) / roll_p95.replace(0, np.nan)

        return df

    def calculate_trend_features(self, df):
        """
        Tendencia (5min/15min), velocidad y aceleración — por métrica de
        latencia (peer/dns1/dns2). A CYCLE_INTERVAL=30s: ventana de
        TREND_WINDOW_SHORT=10 ciclos ≈ 5 min, TREND_WINDOW_LONG=30 ≈ 15 min.
        """
        if df.empty:
            return df
        logging.info("🔧 Calculando tendencia/velocidad/aceleración por métrica...")
        df = df.sort_values('time').copy()

        for name, col in LATENCY_METRICS:
            df[f'latency_trend_5min_{name}'] = (
                df[col].rolling(window=TREND_WINDOW_SHORT, min_periods=1).mean().diff()
            )
            df[f'latency_trend_15min_{name}'] = (
                df[col].rolling(window=TREND_WINDOW_LONG, min_periods=1).mean().diff()
            )
            df[f'latency_velocity_{name}'] = df[col].diff()
            df[f'latency_acceleration_{name}'] = df[col].diff().diff()

        # Spikes de pérdida (ya viene agregada en ventana desde bgp_metrics_new,
        # así que un salto ciclo-a-ciclo acá es una segunda derivada de la señal)
        df['loss_spike_dns1'] = (df['dns1_loss_pct'].diff().fillna(0) > LOSS_SPIKE_THRESHOLD_PCT)
        df['loss_spike_dns2'] = (df['dns2_loss_pct'].diff().fillna(0) > LOSS_SPIKE_THRESHOLD_PCT)

        return df

    # ------------------------------------------------------------------
    # Features contextuales (sin cambios de fondo respecto a v1)
    # ------------------------------------------------------------------
    def calculate_contextual_features(self, df):
        if df.empty:
            return df
        logging.info("🔧 Calculando features contextuales...")
        df = df.copy()
        df['hour_of_day'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_business_hours'] = (
            (df['hour_of_day'] >= 9) & (df['hour_of_day'] < 17) & (df['day_of_week'] < 5)
        ).astype(bool)
        df['is_peak_traffic'] = (
            ((df['hour_of_day'] >= 10) & (df['hour_of_day'] < 14)) |
            ((df['hour_of_day'] >= 15) & (df['hour_of_day'] < 18))
        ).astype(bool)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(bool)
        return df

    def calculate_provider_change_context(self, df):
        """
        Contexto de cambios de provider desde bgp_failover_events. A diferencia
        de v1, NO se intenta reconstruir 'alternative_provider_score' — no es
        derivable porque el motor solo mide el provider activo, no ambos en
        paralelo.

        ⚠️ v2.1 — fix de fuga temporal (data leakage). La versión anterior
        calculaba 'provider_changes_last_hour' con NOW() y 'time_since_last_
        change_min' con MAX(time) GLOBAL de bgp_failover_events — un único
        valor aplicado a TODAS las filas del batch, sin importar el timestamp
        de cada una. Como el batch corre DESPUÉS de que ya ocurrieron los
        eventos, cada fila terminaba "viendo" eventos futuros respecto a su
        propio ciclo (evidencia: filas anteriores al primer failover ya
        mostraban provider_changes_last_hour=2). Ahora ambas features se
        calculan POR FILA, usando solo eventos con time < row.time.
        """
        if df.empty:
            return df
        logging.info("🔧 Calculando contexto de cambios de provider (por fila, sin fuga temporal)...")
        df = df.copy()

        try:
            cur = self.conn.cursor()
            cur.execute("SELECT time FROM bgp_failover_events ORDER BY time")
            event_times = pd.to_datetime(pd.Series([r[0] for r in cur.fetchall()]), utc=True)
            cur.close()
        except Exception as e:
            logging.warning(f"⚠️ Error obteniendo failover_events: {e}")
            event_times = pd.Series([], dtype='datetime64[ns, UTC]')

        if event_times.empty:
            df['provider_changes_last_hour'] = 0
            df['time_since_last_change_min'] = np.nan
            return df

        event_times_np = event_times.values.astype('datetime64[ns]')

        def _row_context(row_time):
            row_time_np = np.datetime64(row_time.tz_convert('UTC').tz_localize(None))
            window_start = row_time_np - np.timedelta64(1, 'h')
            # Eventos estrictamente anteriores a esta fila (evita ver el propio
            # evento del ciclo actual como "pasado")
            prior_mask = event_times_np < row_time_np
            prior_events = event_times_np[prior_mask]

            changes_last_hour = int(((prior_events >= window_start)).sum())
            if len(prior_events) == 0:
                time_since_min = np.nan   # sin evento previo -> NULL, no 0
            else:
                last_change = prior_events.max()
                time_since_min = (row_time_np - last_change) / np.timedelta64(1, 'm')

            return pd.Series({
                'provider_changes_last_hour': changes_last_hour,
                'time_since_last_change_min': time_since_min,
            })

        context = df['time'].apply(_row_context)
        df['provider_changes_last_hour'] = context['provider_changes_last_hour'].astype(int)
        df['time_since_last_change_min'] = context['time_since_last_change_min']
        return df

    # ------------------------------------------------------------------
    # Target multiclase — reemplaza should_failover/failover_event
    # ------------------------------------------------------------------
    def calculate_target(self, df):
        """
        target_decision viene directamente de bgp_metrics_new.decision
        (normal / degradacion / failover / retorno). Las filas con
        decision IN EXCLUDED_DECISIONS se descartan del dataset de
        entrenamiento (no se insertan en ml_features).
        """
        if df.empty:
            return df
        df = df.copy()
        df['target_decision'] = df['decision']
        return df

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def process_and_store(self):
        df = self.load_metrics_incremental()
        if df.empty:
            logging.info("ℹ️ Sin nuevos datos, nada que procesar")
            return 0

        logging.info("🔄 Procesando features (con contexto histórico para rolling stats)...")
        df = self.calculate_zscore_features(df)
        df = self.calculate_cv_p95_features(df)
        df = self.calculate_trend_features(df)
        df = self.calculate_contextual_features(df)
        df = self.calculate_provider_change_context(df)
        df = self.calculate_target(df)

        # Solo se insertan las filas NUEVAS (is_context=False).
        # Las de contexto cumplieron su función (dar historia a las rolling
        # stats) y se descartan aquí.
        df_to_insert = df[~df['is_context']].copy()

        # ✅ Exclusión explícita: 'error' y 'failover_inmediato' no participan
        # del entrenamiento (decisión confirmada en la conversación).
        before = len(df_to_insert)
        df_to_insert = df_to_insert[~df_to_insert['decision'].isin(EXCLUDED_DECISIONS)]
        excluded = before - len(df_to_insert)
        if excluded > 0:
            logging.info(f"🚫 Excluidas {excluded} filas con decision en {EXCLUDED_DECISIONS}")

        if df_to_insert.empty:
            logging.info("ℹ️ Nada para insertar tras filtrar contexto/exclusiones")
            return 0

        # Columnas auxiliares que no van a la tabla
        df_to_insert = df_to_insert.drop(columns=['is_context', 'decision'])

        logging.info("💾 Guardando en ml_features...")
        inserted = 0
        for _, row in df_to_insert.iterrows():
            if self.ts_client.insert_ml_features(row):
                inserted += 1

        logging.info(f"✅ {inserted} registros nuevos grabados en ml_features")
        return inserted


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('/var/log/feature_engine.log')]
    )
    logging.info("=" * 80)
    logging.info("🔧 Feature Engine v2 — Etapa 2 (features temporales por métrica)")
    logging.info("=" * 80)

    engine = FeatureEngineV2()
    logging.info(f"⚙️ Modo: {EXECUTION_MODE.upper()}")
    logging.info(f"⚙️ Ventana corta (tendencia): {TREND_WINDOW_SHORT} ciclos")
    logging.info(f"⚙️ Ventana larga (z-score/CV/p95/tendencia): {TREND_WINDOW_LONG} ciclos")
    logging.info(f"⚙️ Decisiones excluidas del entrenamiento: {EXCLUDED_DECISIONS}")

    inserted = engine.process_and_store()

    logging.info("")
    logging.info("=" * 80)
    logging.info(f"✅ Feature Engine v2 ejecutado — {inserted} registros nuevos")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
