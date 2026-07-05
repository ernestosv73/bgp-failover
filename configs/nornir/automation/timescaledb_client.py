#!/usr/bin/env python3
"""
TimescaleDB Client FINAL con conversión automática de tipos
✅ ACTUALIZADO: Agrega método insert_bgp_metrics_new para nueva tabla
"""
import psycopg2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def sanitize_metrics_dict(metrics: dict) -> dict:
    """
    ✅ Sanitiza un diccionario de métricas convirtiendo tipos numpy a tipos nativos de Python
    """
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer,)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating,)):
            sanitized[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            sanitized[key] = bool(value)
        elif isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


class TimescaleDBClient:
    """Cliente TimescaleDB FINAL con conversión automática de tipos"""
    
    def __init__(self, host='timescaledb', port=5432, database='bgp_failover_db',
                 user='bgp_app', password=None):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password or 'bgp_app_password'
        
        if not self.password:
            raise ValueError("password requerida")
        
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establecer conexión a TimescaleDB"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"✅ Conectado a TimescaleDB en {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Error conectando a TimescaleDB: {e}")
            raise
    
    def health_check(self) -> bool:
        """Verifica que la conexión esté activa"""
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT 1')
            cur.fetchone()
            cur.close()
            return True
        except:
            return False
    
    def insert_bgp_metrics(self, metrics: dict) -> bool:
        """
        Insertar métricas BGP en tabla bgp_metrics (LEGACY)
        ✅ Conversión automática de tipos numpy
        """
        cur = None
        try:
            sanitized_metrics = sanitize_metrics_dict(metrics)
            
            columns = list(sanitized_metrics.keys())
            placeholders = ", ".join(["%s"] * len(columns))
            column_names = ", ".join(columns)
            
            query = f"""
                INSERT INTO bgp_metrics ({column_names})
                VALUES ({placeholders})
            """
            values = [sanitized_metrics[col] for col in columns]
            
            cur = self.conn.cursor()
            cur.execute(query, values)
            self.conn.commit()
            cur.close()
            return True
        except Exception as e:
            if cur:
                cur.close()
            self.conn.rollback()
            logger.error(f"❌ Error insertando en bgp_metrics: {e}")
            return False
    
    def insert_bgp_metrics_new(self, metrics: dict) -> bool:
        """
        ✅ NUEVO: Insertar métricas BGP en tabla bgp_metrics_new
        Nueva estructura con DNS1/DNS2 separados, scores y umbrales
        
        Campos esperados:
        ─────────────────────────────────────────────
        IDENTIFICACIÓN:
            time, provider, peer_ip, peer_asn, cycle_number, host
        
        MÉTRICAS DEL PEER:
            peer_latency_ms, peer_jitter_ms, peer_loss_pct
        
        MÉTRICAS DNS1 (2001:db8:8888::100):
            dns1_latency_ms, dns1_jitter_ms, dns1_loss_pct
        
        MÉTRICAS DNS2 (2001:db8:4444::100):
            dns2_latency_ms, dns2_jitter_ms, dns2_loss_pct
        
        SCORES:
            score, score_dns1, score_dns2, max_score
        
        UMBRALES:
            umbral_failover, umbral_retorno
        
        ESTADO:
            current_provider, provider_changed, provider_change_reason,
            degradation_cycle, sustained_degradation, quality_status, decision
        
        ANÁLISIS FUTURO:
            z_score_peer, z_score_severity,
            rolling_mean, rolling_std, rolling_p95
        ─────────────────────────────────────────────
        """
        cur = None
        try:
            sanitized_metrics = sanitize_metrics_dict(metrics)
            
            columns = list(sanitized_metrics.keys())
            placeholders = ", ".join(["%s"] * len(columns))
            column_names = ", ".join(columns)
            
            query = f"""
                INSERT INTO bgp_metrics_new ({column_names})
                VALUES ({placeholders})
            """
            values = [sanitized_metrics[col] for col in columns]
            
            cur = self.conn.cursor()
            cur.execute(query, values)
            self.conn.commit()
            cur.close()
            
            logger.debug(f"✅ Métricas insertadas en bgp_metrics_new (ciclo {metrics.get('cycle_number', '?')})")
            return True
        except Exception as e:
            if cur:
                cur.close()
            self.conn.rollback()
            logger.error(f"❌ Error insertando en bgp_metrics_new: {e}")
            return False
    
    def insert_failover_event(self, event: dict) -> bool:
        """Inserta un evento de failover"""
        cur = None
        try:
            sanitized_event = sanitize_metrics_dict(event)
            
            columns = list(sanitized_event.keys())
            placeholders = ", ".join(["%s"] * len(columns))
            column_names = ", ".join(columns)
            
            query = f"""
                INSERT INTO bgp_failover_events ({column_names})
                VALUES ({placeholders})
            """
            values = [sanitized_event[col] for col in columns]
            
            cur = self.conn.cursor()
            cur.execute(query, values)
            self.conn.commit()
            cur.close()
            return True
        except Exception as e:
            if cur:
                cur.close()
            self.conn.rollback()
            logger.error(f"❌ Error insertando evento de failover: {e}")
            return False
    
    def insert_ml_features(self, row) -> bool:
        """Inserta un registro de features en ml_features"""
        cur = None
        try:
            if hasattr(row, 'to_dict'):
                metrics = row.to_dict()
            else:
                metrics = dict(row)
            
            sanitized_metrics = sanitize_metrics_dict(metrics)
            
            columns = list(sanitized_metrics.keys())
            placeholders = ", ".join(["%s"] * len(columns))
            column_names = ", ".join(columns)
            
            query = f"""
                INSERT INTO ml_features ({column_names})
                VALUES ({placeholders})
            """
            values = [sanitized_metrics[col] for col in columns]
            
            cur = self.conn.cursor()
            cur.execute(query, values)
            self.conn.commit()
            cur.close()
            return True
        except Exception as e:
            if cur:
                cur.close()
            self.conn.rollback()
            logger.error(f"❌ Error insertando en ml_features: {e}")
            return False
    
    def get_last_cycle_number(self, table='bgp_metrics_new') -> int:
        """Lee el último cycle_number de la tabla especificada"""
        try:
            cur = self.conn.cursor()
            cur.execute(f"SELECT COALESCE(MAX(cycle_number), 0) FROM {table}")
            last_cycle = cur.fetchone()[0]
            cur.close()
            return last_cycle
        except Exception as e:
            logger.error(f"⚠️ Error leyendo cycle_number de {table}: {e}")
            return 0
    
    def get_last_feature_timestamp(self):
        """Lee el último timestamp de ml_features"""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT COALESCE(MAX(time), NULL)
                FROM ml_features
            """)
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"⚠️ Error leyendo last_timestamp: {e}")
            return None
    
    def close(self):
        """Cerrar conexión"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("🔒 Conexión a TimescaleDB cerrada")


if __name__ == '__main__':
    import os
    
    password = os.environ.get('TIMESCALEDB_PASSWORD', 'bgp_app_password')
    
    client = TimescaleDBClient(
        host='timescaledb',
        database='bgp_failover_db',
        user='bgp_app',
        password=password
    )
    
    if not client.health_check():
        logger.error("No se pudo conectar")
        exit(1)
    
    logger.info("✅ Conexión verificada")
    
    # Probar insert_bgp_metrics_new
    test_metric = {
        'time': datetime.now(),
        'provider': 'PROVIDER1',
        'cycle_number': 999999,
        'peer_latency_ms': 5.0,
        'peer_jitter_ms': 1.0,
        'peer_loss_pct': 0.0,
        'dns1_latency_ms': 10.0,
        'dns1_jitter_ms': 2.0,
        'dns1_loss_pct': 0.0,
        'dns2_latency_ms': 15.0,
        'dns2_jitter_ms': 3.0,
        'dns2_loss_pct': 0.0,
        'score': 12.0,
        'score_dns1': 8.0,
        'score_dns2': 12.0,
        'max_score': 12.0,
        'umbral_failover': 11.0,
        'umbral_retorno': 8.0,
        'current_provider': 'PROVIDER1',
        'provider_changed': False,
        'degradation_cycle': 0,
        'sustained_degradation': False,
        'quality_status': 'excellent',
        'decision': 'normal'
    }
    
    result = client.insert_bgp_metrics_new(test_metric)
    if result:
        logger.info("✅ Test insert_bgp_metrics_new: EXITOSO")
        # Limpiar dato de prueba
        try:
            cur = client.conn.cursor()
            cur.execute("DELETE FROM bgp_metrics_new WHERE cycle_number = 999999")
            client.conn.commit()
            cur.close()
            logger.info("🧹 Dato de prueba eliminado")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo eliminar dato de prueba: {e}")
    else:
        logger.error("❌ Test insert_bgp_metrics_new: FALLÓ")
    
    client.close()
