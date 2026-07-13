#!/usr/bin/env python3
"""
link_health_feature_engine.py — Deriva link_health_features desde link_health_metrics
═══════════════════════════════════════════════════════════════════════════════
Versión reducida y adaptada de feature_engine_incremental.py, para el
pipeline de salud de enlace (independiente del motor de failover).

Diferencias respecto al feature engine del motor:
├─ Fuente: link_health_metrics (no bgp_metrics_new) — sin columna 'provider'
│   (un solo enlace, no hay dual-provider).
├─ NO calcula dns1_norm/dns2_norm/jitter*_norm/loss*_norm — esas features
│   normalizan contra un umbral FIJO en ms, y este pipeline busca ESTIMAR
│   ese umbral a partir del modelo, no dárselo como dato de entrada (ver
│   conversación / migration_link_health.sql).
├─ NO calcula nada de 'peer' — no se monitorea acá.
├─ Mismo algoritmo de rolling stats (z-score, CV, p95_dev, z_deriv,
│   tendencia/velocidad/aceleración) sobre dns1/dns2, ventanas idénticas
│   (TREND_WINDOW_SHORT=10, TREND_WINDOW_LONG=30).
└─ Mismo cold-start (tabla completa si link_health_features está vacía) +
   modo incremental con ventana de contexto, igual que el feature engine
   del motor.

USO:
    python3 link_health_feature_engine.py
═══════════════════════════════════════════════════════════════════════════════
"""
import logging

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TREND_WINDOW_SHORT = 10   # ~7.5 min a 45s/ciclo
TREND_WINDOW_LONG = 30    # ~22.5 min a 45s/ciclo
LOSS_SPIKE_THRESHOLD_PCT = 5.0

ZSCORE_METRICS = [
    ('dns1', 'dns1_latency_ms'), ('dns2', 'dns2_latency_ms'),
    ('jitter1', 'dns1_jitter_ms'), ('jitter2', 'dns2_jitter_ms'),
    ('loss1', 'dns1_loss_pct'), ('loss2', 'dns2_loss_pct'),
]
CV_P95_METRICS = [('dns1', 'dns1_latency_ms'), ('dns2', 'dns2_latency_ms')]
LATENCY_METRICS = [('dns1', 'dns1_latency_ms'), ('dns2', 'dns2_latency_ms')]

RAW_COLUMNS = [
    'time', 'cycle_number',
    'dns1_latency_ms', 'dns1_jitter_ms', 'dns1_loss_pct',
    'dns2_latency_ms', 'dns2_jitter_ms', 'dns2_loss_pct',
]


class LinkHealthFeatureEngine:
    def __init__(self, db_host='timescaledb', db_port=5432, db_name='bgp_failover_db',
                 db_user='bgp_app', db_password='bgp_app_password'):
        self.conn = psycopg2.connect(host=db_host, port=db_port, database=db_name,
                                      user=db_user, password=db_password)
        logging.info(f"✅ Conectado a TimescaleDB en {db_host}:{db_port}")

    def get_last_feature_timestamp(self):
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT MAX(time) FROM link_health_features")
            result = cur.fetchone()
            cur.close()
            if result and result[0] is not None:
                return result[0]
            logging.info("ℹ️ link_health_features vacía, se procesará link_health_metrics completa")
            return None
        except Exception as e:
            logging.warning(f"⚠️ No se pudo leer último timestamp: {e}")
            # ⚠️ CRÍTICO: en PostgreSQL, un error dentro de una transacción la
            # deja "abortada" — cualquier consulta posterior en la MISMA
            # conexión falla con "current transaction is aborted", aunque
            # sea sobre una tabla completamente distinta y sana. Sin este
            # rollback, un error aislado acá (ej. permisos en una sola
            # tabla) inutiliza el resto de la corrida — es justo lo que pasó:
            # el error fue en link_health_features, pero tumbó también la
            # lectura de link_health_metrics que le siguió.
            try:
                self.conn.rollback()
            except Exception:
                pass
            return None

    def load_metrics_incremental(self):
        last_timestamp = self.get_last_feature_timestamp()
        is_cold_start = last_timestamp is None

        cols = ', '.join(RAW_COLUMNS)
        if is_cold_start:
            logging.info("📥 Cold-start: cargando link_health_metrics COMPLETA...")
            query = f"SELECT {cols} FROM link_health_metrics ORDER BY time"
        else:
            logging.info(f"📥 Cargando datos después de: {last_timestamp}")
            query = f"SELECT {cols} FROM link_health_metrics WHERE time > '{last_timestamp}'::timestamptz ORDER BY time"

        try:
            df_new = pd.read_sql(query, self.conn)
            df_new['time'] = pd.to_datetime(df_new['time'], utc=True)
        except Exception as e:
            logging.error(f"❌ Error cargando link_health_metrics: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            return pd.DataFrame()

        if df_new.empty:
            logging.info("ℹ️ No hay datos nuevos para procesar")
            return df_new

        df_new['is_context'] = False

        if is_cold_start:
            logging.info(f"✅ Cargados {len(df_new)} registros (tabla completa)")
            df_context = pd.DataFrame()
        else:
            logging.info(f"✅ Cargados {len(df_new)} registros nuevos")
            context_needed = TREND_WINDOW_LONG - 1
            if context_needed > 0:
                first_new_time = df_new['time'].min()
                context_query = (
                    f"SELECT {cols} FROM link_health_metrics "
                    f"WHERE time < '{first_new_time}'::timestamptz ORDER BY time DESC LIMIT {context_needed}"
                )
                try:
                    df_context = pd.read_sql(context_query, self.conn)
                    df_context['time'] = pd.to_datetime(df_context['time'], utc=True)
                    df_context['is_context'] = True
                    logging.info(f"📎 Contexto histórico agregado: {len(df_context)} filas previas")
                except Exception as e:
                    logging.warning(f"⚠️ No se pudo cargar contexto histórico: {e}")
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                    df_context = pd.DataFrame()
            else:
                df_context = pd.DataFrame()

        df = pd.concat([df_context, df_new], ignore_index=True).sort_values('time').reset_index(drop=True)
        return df

    def calculate_zscore_features(self, df):
        for name, col in ZSCORE_METRICS:
            roll_mean = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).mean()
            roll_std = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).std(ddof=0)
            df[f'z_score_{name}'] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        return df

    def calculate_cv_p95_features(self, df):
        for name, col in CV_P95_METRICS:
            roll_mean = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).mean()
            roll_std = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).std(ddof=0)
            roll_p95 = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=3).quantile(0.95)
            df[f'cv_{name}'] = (roll_std / roll_mean.replace(0, np.nan)) * 100
            df[f'p95_dev_{name}'] = (df[col] - roll_p95) / roll_p95.replace(0, np.nan)
            df[f'z_deriv_{name}'] = df[f'z_score_{name}'].diff() if f'z_score_{name}' in df.columns else np.nan
        return df

    def calculate_trend_features(self, df):
        for name, col in LATENCY_METRICS:
            df[f'latency_trend_5min_{name}'] = df[col].rolling(window=TREND_WINDOW_SHORT, min_periods=1).mean().diff()
            df[f'latency_trend_15min_{name}'] = df[col].rolling(window=TREND_WINDOW_LONG, min_periods=1).mean().diff()
            df[f'latency_velocity_{name}'] = df[col].diff()
            df[f'latency_acceleration_{name}'] = df[col].diff().diff()

        df['loss_spike_dns1'] = (df['dns1_loss_pct'].diff().fillna(0) > LOSS_SPIKE_THRESHOLD_PCT)
        df['loss_spike_dns2'] = (df['dns2_loss_pct'].diff().fillna(0) > LOSS_SPIKE_THRESHOLD_PCT)
        return df

    def calculate_contextual_features(self, df):
        """
        ✅ v2 — esquema FCC (paper "Where Has the Time Gone? Examining Over
        a Decade of Broadband Latency Measurements"), reemplaza a
        is_business_hours/is_peak_traffic/is_weekend.

        time_category: 'peak' (19-23h*, solo días de semana) / 'off_peak'
        (resto, días de semana) / 'satsun' (todo el fin de semana).
        *19≤hora<23 = horas 19,20,21,22 ("7PM-11PM") — coincide EXACTAMENTE
        con la ventana que usa el generador sintético por default
        (peak_hour_start=19, peak_hour_duration=4). La versión anterior de
        is_peak_traffic incluía además la hora 23 por error — se corrige acá.

        daypart: resolución más fina y ASIMÉTRICA, más fina cerca del
        horario pico (3 franjas de 2h entre 18-24h) que en el resto del día
        (3 franjas de 6h). Fin de semana: solo 2 mitades de 12h — el paper
        no le da la misma granularidad que a los días de semana.
        """
        df['hour_of_day'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek  # 0=lunes ... 6=domingo
        is_weekend = df['day_of_week'] >= 5

        is_peak_hour = df['hour_of_day'].between(19, 22)  # 19,20,21,22 — "7PM-11PM"
        df['time_category'] = np.where(
            is_weekend, 'satsun', np.where(is_peak_hour, 'peak', 'off_peak')
        )

        weekday_bins = [0, 6, 12, 18, 20, 22, 24]
        weekday_labels = ['00-06', '06-12', '12-18', '18-20', '20-22', '22-24']
        weekend_bins = [0, 12, 24]
        weekend_labels = ['00-12', '12-24']

        daypart_weekday = pd.cut(df['hour_of_day'], bins=weekday_bins, labels=weekday_labels,
                                  right=False, include_lowest=True)
        daypart_weekend = pd.cut(df['hour_of_day'], bins=weekend_bins, labels=weekend_labels,
                                  right=False, include_lowest=True)
        df['daypart'] = np.where(is_weekend, daypart_weekend.astype(str), daypart_weekday.astype(str))

        return df

    def save_features(self, df):
        df_to_save = df[df['is_context'] == False].copy()
        if df_to_save.empty:
            logging.info("ℹ️ No hay filas nuevas para guardar (todo era contexto)")
            return 0

        feature_cols = [
            'time', 'cycle_number',
            'dns1_latency_ms', 'dns1_jitter_ms', 'dns1_loss_pct',
            'dns2_latency_ms', 'dns2_jitter_ms', 'dns2_loss_pct',
            'z_score_dns1', 'z_score_dns2', 'z_score_jitter1', 'z_score_jitter2',
            'z_score_loss1', 'z_score_loss2', 'cv_dns1', 'cv_dns2',
            'p95_dev_dns1', 'p95_dev_dns2', 'z_deriv_dns1', 'z_deriv_dns2',
            'latency_trend_5min_dns1', 'latency_trend_5min_dns2',
            'latency_trend_15min_dns1', 'latency_trend_15min_dns2',
            'latency_velocity_dns1', 'latency_velocity_dns2',
            'latency_acceleration_dns1', 'latency_acceleration_dns2',
            'loss_spike_dns1', 'loss_spike_dns2',
            'hour_of_day', 'day_of_week', 'time_category', 'daypart',
        ]
        df_to_save = df_to_save[feature_cols]

        cur = self.conn.cursor()
        cols = list(df_to_save.columns)
        placeholders = ', '.join(['%s'] * len(cols))
        insert_sql = f"INSERT INTO link_health_features ({', '.join(cols)}) VALUES ({placeholders})"

        records = [tuple(None if pd.isna(v) else v for v in row) for row in df_to_save.itertuples(index=False)]
        from psycopg2.extras import execute_batch
        execute_batch(cur, insert_sql, records, page_size=500)
        self.conn.commit()
        cur.close()

        logging.info(f"💾 {len(records)} registros guardados en link_health_features")
        return len(records)

    def process_and_store(self):
        df = self.load_metrics_incremental()
        if df.empty:
            return 0

        logging.info("🔄 Procesando features...")
        df = self.calculate_zscore_features(df)
        df = self.calculate_cv_p95_features(df)
        df = self.calculate_trend_features(df)
        df = self.calculate_contextual_features(df)

        return self.save_features(df)

    def close(self):
        self.conn.close()


def main():
    logging.info("=" * 80)
    logging.info("🔧 Link Health Feature Engine")
    logging.info("=" * 80)

    engine = LinkHealthFeatureEngine()
    try:
        n = engine.process_and_store()
        logging.info("=" * 80)
        logging.info(f"✅ Feature Engine ejecutado — {n} registros nuevos")
        logging.info("=" * 80)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
