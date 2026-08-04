-- ============================================================================
-- create_database_schema.sql
-- Motor de Failover BGP Inteligente — schema completo de bgp_failover_db
-- ============================================================================
-- Crea desde cero las 4 tablas que necesita bgp_failover_engine_new.py para
-- funcionar: provider_config (config inicial, se lee una vez al arrancar),
-- bgp_metrics_new (métricas crudas + score, una fila por ciclo),
-- bgp_failover_events (auditoría de cada cambio de provider), y ml_features
-- (features derivadas para el pipeline de entrenamiento — Etapa 1 y 2).
--
-- Uso — secuencia completa desde cero, con el docker-compose.yml del proyecto
-- (que define POSTGRES_DB: bgp_failover_db):
--   1) Levantar los contenedores. El entrypoint oficial de Postgres crea la
--      base 'bgp_failover_db' automáticamente en el primer arranque (a
--      partir de la variable de entorno POSTGRES_DB) — NO hace falta un
--      CREATE DATABASE manual en este flujo:
--         docker compose up -d
--   2) Correr este script directamente contra esa base ya existente (crea
--      el rol bgp_app si no existe, la extensión TimescaleDB — que la
--      imagen deja lista para usar pero NO habilita sola en bases nuevas,
--      hay que pedirlo por CREATE EXTENSION explícito — y las 4 tablas):
--         psql -h localhost -U postgres -d bgp_failover_db \
--              -f create_database_schema.sql
--      (contraseña de POSTGRES_PASSWORD en el compose; localhost porque el
--      compose publica el puerto 5432 al host)
--
-- ⚠️ Nota para reproducir esto SIN este docker-compose.yml (ej. un Postgres
-- ya existente, sin la variable POSTGRES_DB): ahí sí hace falta crear la
-- base a mano ANTES del paso 2, ya que CREATE DATABASE no puede ejecutarse
-- dentro de un bloque de transacción BEGIN/COMMIT como el resto de este
-- script:
--         psql -h <host> -U <superusuario> -d postgres \
--              -c "CREATE DATABASE bgp_failover_db;"
--
-- Requiere la extensión TimescaleDB disponible en el servidor de Postgres.
-- ============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- 0) Rol de aplicación — bgp_app
-- ============================================================================
-- ⚠️ Esto faltaba en la primera versión de este script — el resto del
-- archivo ya asumía que 'bgp_app' existía (los GRANT del final apuntan a
-- ese rol), pero nunca se creaba. CREATE ROLE sí puede ir dentro de esta
-- misma transacción (a diferencia de CREATE DATABASE, que NO puede
-- ejecutarse dentro de un bloque BEGIN/COMMIT — ver nota de bootstrap más
-- abajo). Envuelto en un bloque DO para que sea idempotente: CREATE ROLE
-- no soporta "IF NOT EXISTS" de forma nativa en PostgreSQL.
--
-- ⚠️ Contraseña de ejemplo (la usada en el resto de los scripts del
-- proyecto) — cambiarla en un despliegue real, no dejarla como está acá.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'bgp_app') THEN
        CREATE ROLE bgp_app WITH LOGIN PASSWORD 'bgp_app_password';
    END IF;
END
$$;

-- ============================================================================
-- 1) provider_config — configuración inicial de cada provider
-- ============================================================================
-- ⚠️ Tabla NO cubierta por los 3 schemas que se compartieron para este script
-- (bgp_metrics_new / bgp_failover_events / ml_features) — se infirió su
-- estructura a partir de un export CSV de sus datos (provider_config.csv),
-- no de una consulta de schema directa. Verificar tipos antes de asumirla
-- como definitiva, en particular las columnas numéricas (sla_*, *_weight,
-- *_multiplier, *_threshold, switch_margin), que podrían ser NUMERIC en vez
-- de DOUBLE PRECISION según cómo se hayan definido originalmente.
--
-- El motor SOLO lee peer_ip/peer_asn de acá (_load_provider_config()) — el
-- resto de las columnas (sla_*, *_weight, sustained_degradation_cycles,
-- etc.) hoy no están conectadas a ninguna lógica del motor (esos valores
-- siguen viviendo como constantes Python separadas) — ver conversación.
CREATE TABLE IF NOT EXISTS provider_config (
    provider                          TEXT PRIMARY KEY,
    peer_ip                           INET NOT NULL,
    peer_asn                          INTEGER,
    dns_test_ip                       INET,
    dns_test_description              TEXT,
    sla_latency_target_ms             DOUBLE PRECISION,
    sla_jitter_target_ms              DOUBLE PRECISION,
    sla_loss_target_pct               DOUBLE PRECISION,
    sla_uptime_target_pct             DOUBLE PRECISION,
    latency_weight                    DOUBLE PRECISION,
    dns_latency_weight                DOUBLE PRECISION,
    loss_penalty_multiplier           DOUBLE PRECISION,
    jitter_penalty_multiplier         DOUBLE PRECISION,
    sustained_degradation_cycles      INTEGER,
    immediate_failover_loss_threshold DOUBLE PRECISION,
    switch_margin                     DOUBLE PRECISION,
    is_active                         BOOLEAN DEFAULT TRUE,
    created_at                        TIMESTAMPTZ DEFAULT NOW(),
    updated_at                        TIMESTAMPTZ DEFAULT NOW()
);

-- Valores iniciales — AJUSTAR peer_ip/peer_asn a la topología real antes de
-- correr el motor (ver conversación: esta tabla es la fuente de verdad para
-- el peer_ip/peer_asn que se GRABA en cada fila de bgp_metrics_new; el
-- motor mide contra PEER_IPS de bgp_failover_engine_new.py, un valor
-- Python separado — mantener ambos sincronizados a mano).
INSERT INTO provider_config (
    provider, peer_ip, peer_asn, dns_test_ip, dns_test_description,
    sla_latency_target_ms, sla_jitter_target_ms, sla_loss_target_pct, sla_uptime_target_pct,
    latency_weight, dns_latency_weight, loss_penalty_multiplier, jitter_penalty_multiplier,
    sustained_degradation_cycles, immediate_failover_loss_threshold, switch_margin, is_active
) VALUES
    ('PROVIDER1', '2001:db8:1::2', 65002, '2001:db8:700::53', 'PROVIDER1 DNS Endpoint',
     25, 10, 0.5, 99.9, 0.7, 0.3, 100, 0.5, 3, 20, 5, TRUE),
    ('PROVIDER2', '2001:db8:2::2', 65003, '2001:db8:800::53', 'PROVIDER2 DNS Endpoint',
     25, 10, 0.5, 99.9, 0.7, 0.3, 100, 0.5, 3, 20, 5, TRUE)
ON CONFLICT (provider) DO NOTHING;

-- ============================================================================
-- 2) bgp_metrics_new — métricas crudas + score, una fila por ciclo
-- ============================================================================
CREATE TABLE IF NOT EXISTS bgp_metrics_new (
    time                          TIMESTAMPTZ NOT NULL,
    provider                      TEXT NOT NULL,
    peer_ip                       INET,
    peer_asn                      INTEGER,
    cycle_number                  BIGINT NOT NULL,
    host                          TEXT,

    peer_latency_ms               DOUBLE PRECISION NOT NULL,
    peer_jitter_ms                DOUBLE PRECISION NOT NULL,
    peer_loss_pct                 DOUBLE PRECISION NOT NULL,
    dns1_latency_ms                DOUBLE PRECISION,
    dns1_jitter_ms                 DOUBLE PRECISION,
    dns1_loss_pct                  DOUBLE PRECISION,
    dns2_latency_ms                DOUBLE PRECISION,
    dns2_jitter_ms                 DOUBLE PRECISION,
    dns2_loss_pct                  DOUBLE PRECISION,

    score                          DOUBLE PRECISION,
    score_dns1                     DOUBLE PRECISION,
    score_dns2                     DOUBLE PRECISION,
    max_score                      DOUBLE PRECISION,
    umbral_failover                DOUBLE PRECISION,
    umbral_retorno                 DOUBLE PRECISION,

    current_provider               TEXT,
    provider_changed               BOOLEAN,
    provider_change_reason         TEXT,
    degradation_cycle              INTEGER,
    quality_status                 TEXT,
    decision                       VARCHAR(20),

    peer_norm                      DOUBLE PRECISION,
    dns1_norm                      DOUBLE PRECISION,
    dns2_norm                      DOUBLE PRECISION,
    jitter1_norm                   DOUBLE PRECISION,
    jitter2_norm                   DOUBLE PRECISION,
    loss1_window_pct               DOUBLE PRECISION,
    loss2_window_pct               DOUBLE PRECISION,
    loss1_norm                     DOUBLE PRECISION,
    loss2_norm                     DOUBLE PRECISION,

    base_score_dns1                DOUBLE PRECISION,
    base_score_dns2                DOUBLE PRECISION,
    severity_multiplier_dns1       DOUBLE PRECISION,
    severity_multiplier_dns2       DOUBLE PRECISION,

    weight_peer                    DOUBLE PRECISION,
    weight_dns                     DOUBLE PRECISION,
    weight_jitter                  DOUBLE PRECISION,
    cap_value                      DOUBLE PRECISION,
    loss_sla_pct                   DOUBLE PRECISION,
    immediate_failover_triggered   BOOLEAN,

    jitter1_window_ms              DOUBLE PRECISION,
    jitter2_window_ms              DOUBLE PRECISION
);

SELECT create_hypertable('bgp_metrics_new', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_bgp_metrics_new_cycle ON bgp_metrics_new (cycle_number);
CREATE INDEX IF NOT EXISTS idx_bgp_metrics_new_provider ON bgp_metrics_new (current_provider, time DESC);

-- ============================================================================
-- 3) bgp_failover_events — auditoría de cada cambio de provider
-- ============================================================================
CREATE TABLE IF NOT EXISTS bgp_failover_events (
    event_id                 BIGINT GENERATED ALWAYS AS IDENTITY,
    time                     TIMESTAMPTZ NOT NULL,
    previous_provider        TEXT NOT NULL,
    new_provider              TEXT NOT NULL,
    change_reason             TEXT,
    previous_provider_score   DOUBLE PRECISION,
    new_provider_score        DOUBLE PRECISION,
    score_improvement         DOUBLE PRECISION,
    detected_by                TEXT,
    detection_cycles           INTEGER,
    success                    BOOLEAN,
    rollback                   BOOLEAN,
    notes                      TEXT
);

SELECT create_hypertable('bgp_failover_events', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_bgp_failover_events_time ON bgp_failover_events (time DESC);

-- ============================================================================
-- 4) ml_features — features derivadas (Etapa 1 + Etapa 2) para entrenamiento
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_features (
    time                            TIMESTAMPTZ NOT NULL,
    provider                        TEXT NOT NULL,
    cycle_number                    INTEGER,

    peer_latency_ms                 DOUBLE PRECISION,
    peer_loss_pct                   DOUBLE PRECISION,
    peer_jitter_ms                  DOUBLE PRECISION,
    dns1_latency_ms                 DOUBLE PRECISION,
    dns1_jitter_ms                  DOUBLE PRECISION,
    dns1_loss_pct                   DOUBLE PRECISION,
    dns2_latency_ms                 DOUBLE PRECISION,
    dns2_jitter_ms                  DOUBLE PRECISION,
    dns2_loss_pct                   DOUBLE PRECISION,

    hour_of_day                     INTEGER,
    day_of_week                     INTEGER,
    is_business_hours               BOOLEAN,
    is_peak_traffic                 BOOLEAN,
    is_weekend                      BOOLEAN,
    provider_changes_last_hour      INTEGER,
    time_since_last_change_min      DOUBLE PRECISION,

    degradation_cycle               INTEGER,
    provider_changed                BOOLEAN,

    peer_norm                       DOUBLE PRECISION,
    dns1_norm                       DOUBLE PRECISION,
    dns2_norm                       DOUBLE PRECISION,
    jitter1_norm                    DOUBLE PRECISION,
    jitter2_norm                    DOUBLE PRECISION,
    loss1_norm                      DOUBLE PRECISION,
    loss2_norm                      DOUBLE PRECISION,

    base_score_dns1                 DOUBLE PRECISION,
    base_score_dns2                 DOUBLE PRECISION,
    severity_multiplier_dns1        DOUBLE PRECISION,
    severity_multiplier_dns2        DOUBLE PRECISION,
    score_dns1                      DOUBLE PRECISION,
    score_dns2                      DOUBLE PRECISION,
    max_score                       DOUBLE PRECISION,
    quality_status                  VARCHAR(20),

    z_score_peer                    DOUBLE PRECISION,
    z_score_dns1                    DOUBLE PRECISION,
    z_score_dns2                    DOUBLE PRECISION,
    z_score_jitter1                 DOUBLE PRECISION,
    z_score_jitter2                 DOUBLE PRECISION,
    z_score_loss1                   DOUBLE PRECISION,
    z_score_loss2                   DOUBLE PRECISION,

    cv_peer                         DOUBLE PRECISION,
    cv_dns1                         DOUBLE PRECISION,
    cv_dns2                         DOUBLE PRECISION,

    p95_dev_peer                    DOUBLE PRECISION,
    p95_dev_dns1                    DOUBLE PRECISION,
    p95_dev_dns2                    DOUBLE PRECISION,

    z_deriv_peer                    DOUBLE PRECISION,
    z_deriv_dns1                    DOUBLE PRECISION,
    z_deriv_dns2                    DOUBLE PRECISION,

    latency_trend_5min_peer         DOUBLE PRECISION,
    latency_trend_5min_dns1         DOUBLE PRECISION,
    latency_trend_5min_dns2         DOUBLE PRECISION,
    latency_trend_15min_peer        DOUBLE PRECISION,
    latency_trend_15min_dns1        DOUBLE PRECISION,
    latency_trend_15min_dns2        DOUBLE PRECISION,
    latency_velocity_peer           DOUBLE PRECISION,
    latency_velocity_dns1           DOUBLE PRECISION,
    latency_velocity_dns2           DOUBLE PRECISION,
    latency_acceleration_peer       DOUBLE PRECISION,
    latency_acceleration_dns1       DOUBLE PRECISION,
    latency_acceleration_dns2       DOUBLE PRECISION,

    loss_spike_dns1                 BOOLEAN,
    loss_spike_dns2                 BOOLEAN,

    target_decision                 VARCHAR(20)
);

SELECT create_hypertable('ml_features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ml_features_cycle ON ml_features (cycle_number);

COMMIT;

-- ============================================================================
-- Permisos — ajustar el nombre de rol si no es 'bgp_app'. Ver conversación:
-- una migración sin GRANT explícito puede dejar alguna tabla inaccesible
-- para el usuario de aplicación según qué rol la haya creado, aunque las
-- demás sí funcionen (pasó con link_health_features en su momento).
-- ============================================================================
GRANT SELECT, INSERT, UPDATE, DELETE ON provider_config TO bgp_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON bgp_metrics_new TO bgp_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON bgp_failover_events TO bgp_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ml_features TO bgp_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bgp_app;

-- ============================================================================
-- Notas:
-- 1. Este script cubre el pipeline PRINCIPAL (motor de failover +
--    entrenamiento vía target_decision). El pipeline paralelo de detección
--    de anomalías (link_health_metrics/link_health_ground_truth/
--    link_health_features) tiene sus propias migraciones separadas
--    (migration_link_health.sql + migration_link_health_fcc_time.sql) —
--    agregar si ese pipeline también necesita ser reproducible.
-- 2. No se incluye configuración de compresión TimescaleDB (observada como
--    activa en bgp_metrics_new de la instancia actual) — agregar
--    manualmente si se quiere replicar ese aspecto exactamente.
-- ============================================================================
