-- 1. Crear la tabla base para los resultados de ML
CREATE TABLE IF NOT EXISTS link_health_metrics_ml_results (
    time TIMESTAMPTZ PRIMARY KEY,
    dns1_latency_ms DOUBLE PRECISION,
    dns2_latency_ms DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    is_anomaly INTEGER
);

-- 2. Convertir en Hipertabla de TimescaleDB (particionada por tiempo)
SELECT create_hypertable('link_health_metrics_ml_results', 'time', if_not_exists => TRUE);

-- 3. Crear índice para consultas rápidas en Grafana
CREATE INDEX IF NOT EXISTS idx_link_health_metrics_ml_results_time ON link_health_metrics_ml_results (time DESC);

-- 4. Otorgar permisos al usuario de la aplicación
GRANT SELECT, INSERT, UPDATE, DELETE ON link_health_metrics_ml_results TO bgp_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bgp_app;
