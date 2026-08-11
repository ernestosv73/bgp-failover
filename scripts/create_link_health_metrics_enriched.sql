-- 1. Crear la tabla base
CREATE TABLE IF NOT EXISTS link_health_metrics_enriched (
    time TIMESTAMPTZ PRIMARY KEY,
    cycle_number BIGINT,
    dns1_latency_ms DOUBLE PRECISION,
    dns1_jitter_ms DOUBLE PRECISION,
    dns1_loss_pct DOUBLE PRECISION,
    dns2_latency_ms DOUBLE PRECISION,
    dns2_jitter_ms DOUBLE PRECISION,
    dns2_loss_pct DOUBLE PRECISION,
    hour INTEGER,
    day_of_week INTEGER,
    is_business_hours INTEGER,
    dns1_latency_rolling_mean DOUBLE PRECISION,
    dns1_latency_rolling_std DOUBLE PRECISION,
    dns1_jitter_rolling_mean DOUBLE PRECISION,
    dns2_latency_rolling_mean DOUBLE PRECISION,
    dns2_latency_rolling_std DOUBLE PRECISION,
    dns2_jitter_rolling_mean DOUBLE PRECISION
);

-- 2. Convertir la tabla en una Hipertabla de TimescaleDB (particionada por tiempo)
SELECT create_hypertable('link_health_metrics_enriched', 'time', if_not_exists => TRUE);

-- 3. Crear índices para optimizar consultas de series temporales y ML
CREATE INDEX IF NOT EXISTS idx_link_health_metrics_enriched_time ON link_health_metrics_enriched (time DESC);
CREATE INDEX IF NOT EXISTS idx_link_health_metrics_enriched_cycle ON link_health_metrics_enriched (cycle_number);

-- 4. Otorgar permisos al usuario de la aplicación (bgp_app)
GRANT SELECT, INSERT, UPDATE, DELETE ON link_health_metrics_enriched TO bgp_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bgp_app;
