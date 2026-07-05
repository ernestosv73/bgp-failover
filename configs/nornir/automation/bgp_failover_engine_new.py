#!/usr/bin/env python3
"""
BGP Failover Engine - VERSIÓN DNS1/DNS2 CON MTR DOBLE Y FALLBACK
✅ NUEVA LÓGICA:
├─ Monitoreo solo de PROVIDER1 (peer + DNS1 + DNS2)
├─ MTR se ejecuta DOS veces por ciclo:
│   ├─ MTR #1: PROVIDER1 → DNS1 (2001:db8:8888::100)
│   └─ MTR #2: PROVIDER1 → DNS2 (2001:db8:4444::100)
├─ ✅ FALLBACK: Si no se encuentra peer en DNS2, usar peer de DNS1
├─ Score individual por DNS (score_dns1, score_dns2)
├─ Failover si max(score_dns1, score_dns2) > UMBRAL_FAILOVER (11.0)
├─ Retorno si max(score_dns1, score_dns2) < UMBRAL_RETORNO (8.0)
├─ 3 ciclos de degradación sostenida
├─ Failover inmediato si pérdida >= 20%
└─ Sin detección combinada (Z-score, absolute, relative)

✅ SIMPLIFICACIONES:
├─ ❌ Eliminado envío a Elasticsearch
├─ ❌ Eliminado envío a NetBox
├─ ❌ Eliminada detección combinada
├─ ❌ Eliminado switch_margin
├─ ✅ Solo monitoreo de PROVIDER1
├─ ✅ MTR doble (DNS1 + DNS2) con fallback
└─ ✅ Nueva tabla bgp_metrics_new
"""
import time
import logging
import subprocess
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from timescaledb_client import TimescaleDBClient
    TIMESCALEDB_AVAILABLE = True
except ImportError:
    TIMESCALEDB_AVAILABLE = False
    logging.warning("⚠️ TimescaleDB client no disponible")

try:
    from bgp_failover_config import (
        MTR_CONFIG, PEER_IPS, IP_VERSIONS,
        PROVIDERS, CYCLE_INTERVAL, POLICY_RULE_IDS
    )
except ImportError:
    # Configuración por defecto
    MTR_CONFIG = {'count': 5, 'timeout': 30, 'packet_size': 64, 'interval': 0.5}
    PEER_IPS = {
        'PROVIDER1': '2001:db8:ffaa::255',
        'PROVIDER2': '2001:db8:ffac::255'
    }
    IP_VERSIONS = {'PROVIDER1': '6', 'PROVIDER2': '6'}
    PROVIDERS = ['PROVIDER1', 'PROVIDER2']
    CYCLE_INTERVAL = 30
    POLICY_RULE_IDS = {
        'EXPORT-TO-PROVIDER1': 1, 'EXPORT-TO-PROVIDER2': 2,
        'SET-LOCAL-PREF-PROVIDER1': 3, 'SET-LOCAL-PREF-PROVIDER2': 4
    }

# === CONFIGURACIÓN PRINCIPAL ===
TIMESCALEDB_ENABLED = True
TIMESCALEDB_HOST = 'timescaledb'
TIMESCALEDB_PORT = 5432
TIMESCALEDB_DB = 'bgp_failover_db'
TIMESCALEDB_USER = 'bgp_app'
TIMESCALEDB_PASSWORD = 'bgp_app_password'

# === PARÁMETROS DE FAILOVER ===
SUSTAINED_DEGRADATION_CYCLES = 3
IMMEDIATE_FAILOVER_PACKET_LOSS = 20.0

# ✅ UMBRALES DE SCORING (nueva lógica)
UMBRAL_FAILOVER = 11.0   # Si max_score > X durante 3 ciclos → FAILOVER
UMBRAL_RETORNO = 8.0     # Si max_score < Y → RETORNO a PROVIDER1

# ✅ RENOMBRADO: DNS_DESTINATIONS (antes MTR_DESTINATIONS)
DNS_DESTINATIONS = {
    'DNS1': '2001:db8:8888::100',   # Thresholds: warning=15ms, critical=30ms
    'DNS2': '2001:db8:4444::100'    # Thresholds: warning=30ms, critical=60ms
}

# ✅ THRESHOLDS POR DNS (diferentes)
DNS_THRESHOLDS = {
    'DNS1': {  # 2001:db8:8888::100
        'warning': 15.0,
        'critical': 30.0
    },
    'DNS2': {  # 2001:db8:4444::100
        'warning': 30.0,
        'critical': 60.0
    }
}

# Thresholds del peer (PROVIDER1)
PEER_THRESHOLDS = {
    'warning': 12.0,
    'critical': 25.0
}

# ✅ PESOS DE SCORING (alineados con draft IETF)
SCORING_WEIGHTS = {
    'peer_latency': 0.4,
    'dns_latency': 0.6,
    'loss': 0.5,
    'jitter': 0.5
}

ROLLING_HISTORY_SIZE = 10


@dataclass
class LatencyMetrics:
    """Métricas de latencia para PROVIDER1 con DNS1 y DNS2 separados"""
    # Peer (común para ambos DNS)
    peer_avg: float
    peer_loss: float
    peer_stddev: float
    
    # DNS1 (2001:db8:8888::100)
    dns1_avg: float
    dns1_loss: float
    dns1_stddev: float
    
    # DNS2 (2001:db8:4444::100)
    dns2_avg: float
    dns2_loss: float
    dns2_stddev: float
    
    # Scores calculados
    score_dns1: float = 0.0
    score_dns2: float = 0.0
    max_score: float = 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Verifica si el provider está saludable (pérdida < 20%)"""
        return (
            self.peer_loss < IMMEDIATE_FAILOVER_PACKET_LOSS and
            self.dns1_loss < IMMEDIATE_FAILOVER_PACKET_LOSS and
            self.dns2_loss < IMMEDIATE_FAILOVER_PACKET_LOSS
        )
    
    @property
    def has_peer_warning(self) -> bool:
        return self.peer_avg >= PEER_THRESHOLDS['warning']
    
    @property
    def has_peer_critical(self) -> bool:
        return self.peer_avg >= PEER_THRESHOLDS['critical']
    
    @property
    def has_dns1_warning(self) -> bool:
        return self.dns1_avg >= DNS_THRESHOLDS['DNS1']['warning']
    
    @property
    def has_dns1_critical(self) -> bool:
        return self.dns1_avg >= DNS_THRESHOLDS['DNS1']['critical']
    
    @property
    def has_dns2_warning(self) -> bool:
        return self.dns2_avg >= DNS_THRESHOLDS['DNS2']['warning']
    
    @property
    def has_dns2_critical(self) -> bool:
        return self.dns2_avg >= DNS_THRESHOLDS['DNS2']['critical']
    
    @property
    def has_packet_loss(self) -> bool:
        return (
            self.peer_loss > 0.0 or
            self.dns1_loss > 0.0 or
            self.dns2_loss > 0.0
        )
    
    def calculate_scores(self):
        """
        ✅ Calcula scores para DNS1 y DNS2 usando la fórmula:
        score = (peer_avg × 0.4) + (dns_avg × 0.6) + (loss × 0.5) + (jitter × 0.5)
        """
        # Score DNS1
        weighted_latency_dns1 = (
            self.peer_avg * SCORING_WEIGHTS['peer_latency'] +
            self.dns1_avg * SCORING_WEIGHTS['dns_latency']
        )
        loss_penalty_dns1 = (
            (self.peer_loss + self.dns1_loss) / 2 * SCORING_WEIGHTS['loss']
        )
        jitter_penalty_dns1 = (
            (self.peer_stddev + self.dns1_stddev) / 2 * SCORING_WEIGHTS['jitter']
        )
        self.score_dns1 = weighted_latency_dns1 + loss_penalty_dns1 + jitter_penalty_dns1
        
        # Score DNS2
        weighted_latency_dns2 = (
            self.peer_avg * SCORING_WEIGHTS['peer_latency'] +
            self.dns2_avg * SCORING_WEIGHTS['dns_latency']
        )
        loss_penalty_dns2 = (
            (self.peer_loss + self.dns2_loss) / 2 * SCORING_WEIGHTS['loss']
        )
        jitter_penalty_dns2 = (
            (self.peer_stddev + self.dns2_stddev) / 2 * SCORING_WEIGHTS['jitter']
        )
        self.score_dns2 = weighted_latency_dns2 + loss_penalty_dns2 + jitter_penalty_dns2
        
        # Max score (usado para decisión)
        self.max_score = max(self.score_dns1, self.score_dns2)


class BGPFailoverEngine:
    """Motor de failover BGP con monitoreo DNS1/DNS2 (MTR doble con fallback)"""
    
    def __init__(self):
        self.ts_client = None
        self.provider_asn_map = {}
        self.provider_peer_ip_map = {}
        
        # Historial para rolling stats
        self.metrics_history = []
        
        # Estado del failover
        self.cycle_count = 1
        self.current_primary_provider = 'PROVIDER1'
        self.degradation_counter = 0
        
        # Inicializar TimescaleDB
        if TIMESCALEDB_ENABLED and TIMESCALEDB_AVAILABLE:
            try:
                self.ts_client = TimescaleDBClient(
                    host=TIMESCALEDB_HOST,
                    port=TIMESCALEDB_PORT,
                    database=TIMESCALEDB_DB,
                    user=TIMESCALEDB_USER,
                    password=TIMESCALEDB_PASSWORD
                )
                logging.info("✅ TimescaleDB Client inicializado")
                self._load_provider_config()
                self.cycle_count = self._load_last_cycle_number()
                self.current_primary_provider = self._load_current_provider()
            except Exception as e:
                logging.error(f"❌ Error inicializando TimescaleDB: {e}")
                self.ts_client = None
        
        # Log de configuración
        logging.info(f"🚀 Engine inicializado:")
        logging.info(f"   Ciclo actual: {self.cycle_count}")
        logging.info(f"   Provider actual: {self.current_primary_provider}")
        logging.info(f"   🎯 Umbral Failover: {UMBRAL_FAILOVER}")
        logging.info(f"   🎯 Umbral Retorno: {UMBRAL_RETORNO}")
        logging.info(f"   📊 DNS1 ({DNS_DESTINATIONS['DNS1']}): "
                    f"warning={DNS_THRESHOLDS['DNS1']['warning']}ms, "
                    f"critical={DNS_THRESHOLDS['DNS1']['critical']}ms")
        logging.info(f"   📊 DNS2 ({DNS_DESTINATIONS['DNS2']}): "
                    f"warning={DNS_THRESHOLDS['DNS2']['warning']}ms, "
                    f"critical={DNS_THRESHOLDS['DNS2']['critical']}ms")
        logging.info(f"   📊 Peer Thresholds: warning={PEER_THRESHOLDS['warning']}ms, "
                    f"critical={PEER_THRESHOLDS['critical']}ms")
        logging.info(f"   📊 Scoring Weights: {SCORING_WEIGHTS}")
        logging.info(f"   📊 Rolling History: {ROLLING_HISTORY_SIZE} ciclos")
    
    def _load_provider_config(self):
        """Carga provider_asn y peer_ip desde TimescaleDB"""
        try:
            if not self.ts_client or not self.ts_client.conn:
                return
            cur = self.ts_client.conn.cursor()
            cur.execute("SELECT provider, peer_asn, peer_ip FROM provider_config")
            for provider, asn, peer_ip in cur.fetchall():
                self.provider_asn_map[provider] = asn
                self.provider_peer_ip_map[provider] = peer_ip
            cur.close()
            logging.info(f"✅ Configuración de {len(self.provider_asn_map)} providers cargada")
        except Exception as e:
            logging.error(f"❌ Error cargando provider_config: {e}")
    
    def _load_last_cycle_number(self) -> int:
        """Lee el último cycle_number de bgp_metrics_new"""
        try:
            if not self.ts_client or not self.ts_client.conn:
                return 1
            cur = self.ts_client.conn.cursor()
            cur.execute("SELECT COALESCE(MAX(cycle_number), 0) FROM bgp_metrics_new")
            last_cycle = cur.fetchone()[0]
            cur.close()
            next_cycle = last_cycle + 1
            logging.info(f"✅ Último ciclo en BD: {last_cycle} → Próximo: {next_cycle}")
            return next_cycle
        except Exception as e:
            logging.error(f"⚠️ Error leyendo cycle_number: {e}")
            return 1
    
    def _load_current_provider(self) -> str:
        """Lee el provider actual del último failover event"""
        try:
            if not self.ts_client or not self.ts_client.conn:
                return "PROVIDER1"
            cur = self.ts_client.conn.cursor()
            cur.execute("""
                SELECT new_provider
                FROM bgp_failover_events
                ORDER BY event_id DESC
                LIMIT 1
            """)
            result = cur.fetchone()
            cur.close()
            if result:
                provider = result[0]
                logging.info(f"✅ Provider actual en BD: {provider}")
                return provider
            else:
                logging.info("ℹ️ No hay eventos previos, usando PROVIDER1")
                return "PROVIDER1"
        except Exception as e:
            logging.error(f"⚠️ Error leyendo provider actual: {e}")
            return "PROVIDER1"
    
    def run_mtr(self, destination: str, ip_version: str) -> Optional[Dict[str, Any]]:
        """Ejecuta MTR a un destino específico"""
        try:
            cmd = [
                'mtr', f'-{ip_version}', '-n', '-j',
                '-c', str(MTR_CONFIG['count']),
                '-s', str(MTR_CONFIG['packet_size']),
                '-i', str(MTR_CONFIG['interval']),
                destination
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=MTR_CONFIG['timeout']
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logging.warning(f"⚠️ MTR a {destination} falló con código {result.returncode}: {result.stderr}")
                return None
        except Exception as e:
            logging.error(f"❌ Error ejecutando MTR a {destination}: {e}")
            return None
    
    def extract_metrics_single_dns(
        self,
        mtr_report: Dict[str, Any],
        dns_name: str,
        peer_fallback: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, float]]:
        """
        ✅ EXTRAER MÉTRICAS: peer + un DNS específico
        ✅ FALLBACK: Si no se encuentra peer, usar peer_fallback
        
        Args:
            mtr_report: Reporte JSON del MTR
            dns_name: Nombre del DNS (DNS1 o DNS2)
            peer_fallback: Métricas del peer obtenidas de otro MTR (opcional)
        
        Returns:
            Dict con: peer_avg, peer_loss, peer_stddev, dns_avg, dns_loss, dns_stddev
        """
        try:
            hubs = mtr_report['report']['hubs']
            
            peer_ip = PEER_IPS['PROVIDER1']  # Siempre medimos desde PROVIDER1
            dns_ip = DNS_DESTINATIONS[dns_name]
            
            peer_hop = None
            dns_hop = None
            
            for hop in hubs:
                host = hop.get('host')
                if host == peer_ip:
                    peer_hop = hop
                elif host == dns_ip:
                    dns_hop = hop
            
            # ✅ FALLBACK: Si no se encontró peer, usar peer_fallback
            if not peer_hop:
                if peer_fallback:
                    logging.info(
                        f"ℹ️ Peer no encontrado en MTR a {dns_name}, "
                        f"usando peer del MTR a DNS1 (fallback)"
                    )
                    peer_metrics = peer_fallback
                else:
                    logging.warning(f"⚠️ No se encontró hop peer ({peer_ip}) en MTR a {dns_name}")
                    return None
            else:
                peer_metrics = {
                    'peer_avg': float(peer_hop.get('Avg', float('inf'))),
                    'peer_loss': float(peer_hop.get('Loss%', 0.0)),
                    'peer_stddev': float(peer_hop.get('StDev', 0.0))
                }
            
            if not dns_hop:
                logging.warning(f"⚠️ No se encontró hop {dns_name} ({dns_ip})")
                return None
            
            return {
                'peer_avg': peer_metrics['peer_avg'],
                'peer_loss': peer_metrics['peer_loss'],
                'peer_stddev': peer_metrics['peer_stddev'],
                'dns_avg': float(dns_hop.get('Avg', float('inf'))),
                'dns_loss': float(dns_hop.get('Loss%', 0.0)),
                'dns_stddev': float(dns_hop.get('StDev', 0.0))
            }
        except Exception as e:
            logging.error(f"❌ Error extrayendo métricas para {dns_name}: {e}")
            return None
    
    def measure_provider_latency(self) -> Optional[LatencyMetrics]:
        """
        ✅ MÉTODO CORREGIDO: Ejecuta MTR DOS veces con fallback
        1. MTR a DNS1 → extrae peer + DNS1 metrics
        2. MTR a DNS2 → extrae peer + DNS2 metrics (usa peer de DNS1 si no encuentra)
        3. Combina en un solo LatencyMetrics
        """
        ip_version = IP_VERSIONS.get('PROVIDER1', '6')
        
        # === MTR #1: PROVIDER1 → DNS1 ===
        logging.info(f"📡 Ejecutando MTR a DNS1 ({DNS_DESTINATIONS['DNS1']})...")
        mtr_dns1 = self.run_mtr(DNS_DESTINATIONS['DNS1'], ip_version)
        
        if not mtr_dns1:
            logging.error("❌ No se pudo obtener MTR a DNS1")
            return None
        
        metrics_dns1 = self.extract_metrics_single_dns(mtr_dns1, 'DNS1')
        if not metrics_dns1:
            logging.error("❌ No se pudieron extraer métricas de DNS1")
            return None
        
        # ✅ EXTRAER PEER DEL MTR A DNS1 PARA FALLBACK
        peer_from_dns1 = {
            'peer_avg': metrics_dns1['peer_avg'],
            'peer_loss': metrics_dns1['peer_loss'],
            'peer_stddev': metrics_dns1['peer_stddev']
        }
        
        # === MTR #2: PROVIDER1 → DNS2 ===
        logging.info(f"📡 Ejecutando MTR a DNS2 ({DNS_DESTINATIONS['DNS2']})...")
        mtr_dns2 = self.run_mtr(DNS_DESTINATIONS['DNS2'], ip_version)
        
        if not mtr_dns2:
            logging.error("❌ No se pudo obtener MTR a DNS2")
            return None
        
        # ✅ USAR FALLBACK: Pasar peer_from_dns1 como respaldo
        metrics_dns2 = self.extract_metrics_single_dns(
            mtr_dns2, 'DNS2', peer_fallback=peer_from_dns1
        )
        if not metrics_dns2:
            logging.error("❌ No se pudieron extraer métricas de DNS2")
            return None
        
        # === Combinar métricas en un solo LatencyMetrics ===
        # Usamos peer_avg del MTR a DNS1 (más confiable)
        combined_metrics = LatencyMetrics(
            peer_avg=peer_from_dns1['peer_avg'],
            peer_loss=peer_from_dns1['peer_loss'],
            peer_stddev=peer_from_dns1['peer_stddev'],
            dns1_avg=metrics_dns1['dns_avg'],
            dns1_loss=metrics_dns1['dns_loss'],
            dns1_stddev=metrics_dns1['dns_stddev'],
            dns2_avg=metrics_dns2['dns_avg'],
            dns2_loss=metrics_dns2['dns_loss'],
            dns2_stddev=metrics_dns2['dns_stddev']
        )
        
        # Calcular scores
        combined_metrics.calculate_scores()
        
        logging.info(
            f"📊 PROVIDER1 - "
            f"Peer: {combined_metrics.peer_avg:.2f}ms | "
            f"DNS1: {combined_metrics.dns1_avg:.2f}ms | "
            f"DNS2: {combined_metrics.dns2_avg:.2f}ms | "
            f"Score DNS1: {combined_metrics.score_dns1:.2f} | "
            f"Score DNS2: {combined_metrics.score_dns2:.2f} | "
            f"Max: {combined_metrics.max_score:.2f}"
        )
        
        return combined_metrics
    
    def calculate_rolling_stats(self) -> Dict[str, float]:
        """Calcula estadísticas rolling (mean, std, p95) del historial"""
        if len(self.metrics_history) < 3:
            return {
                'mean': 0.0,
                'std': 0.0,
                'p95': 0.0,
                'count': len(self.metrics_history)
            }
        
        # Usar max_score del historial
        scores = [m.max_score for m in self.metrics_history if m.max_score < 900]
        
        if len(scores) < 3:
            return {
                'mean': 0.0,
                'std': 0.0,
                'p95': 0.0,
                'count': len(scores)
            }
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'p95': float(np.percentile(scores, 95)),
            'count': len(scores)
        }
    
    def should_switch_provider(self) -> Tuple[str, str, Optional[LatencyMetrics]]:
        """
        ✅ NUEVA LÓGICA DE FAILOVER SIMPLIFICADA:
        1. Medir PROVIDER1 (peer + DNS1 + DNS2) con MTR DOBLE
        2. Calcular score_dns1 y score_dns2
        3. Si max_score > UMBRAL_FAILOVER durante 3 ciclos → FAILOVER
        4. Si max_score < UMBRAL_RETORNO → RETORNO
        5. Si pérdida >= 20% → FAILOVER inmediato
        """
        # Medir PROVIDER1 con MTR doble
        metrics = self.measure_provider_latency()
        
        if not metrics:
            logging.error("❌ No se pudieron obtener métricas")
            return self.current_primary_provider, "Error de medición", None
        
        # Agregar al historial
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > ROLLING_HISTORY_SIZE:
            self.metrics_history.pop(0)
        
        # Log de estado
        logging.info("📈 Estado actual:")
        logging.info(f"   ⭐ Provider actual: {self.current_primary_provider}")
        logging.info(f"   📊 Max Score: {metrics.max_score:.2f} "
                    f"(DNS1: {metrics.score_dns1:.2f}, DNS2: {metrics.score_dns2:.2f})")
        logging.info(f"   🎯 Umbral Failover: {UMBRAL_FAILOVER} | Umbral Retorno: {UMBRAL_RETORNO}")
        
        # ====================================================================
        # 1️⃣ Verificar pérdida crítica (failover inmediato)
        # ====================================================================
        if not metrics.is_healthy:
            logging.warning(f"⚠️ Pérdida crítica detectada: "
                          f"peer={metrics.peer_loss}%, dns1={metrics.dns1_loss}%, dns2={metrics.dns2_loss}%")
            self.degradation_counter = 0
            
            new_provider = 'PROVIDER2' if self.current_primary_provider == 'PROVIDER1' else 'PROVIDER1'
            return new_provider, f"Pérdida crítica ({max(metrics.peer_loss, metrics.dns1_loss, metrics.dns2_loss)}%)", metrics
        
        # ====================================================================
        # 2️⃣ Si estamos en PROVIDER2: verificar retorno a PROVIDER1
        # ====================================================================
        if self.current_primary_provider == 'PROVIDER2':
            if metrics.max_score < UMBRAL_RETORNO:
                logging.info(f"✅ Retorno a PROVIDER1 (max_score {metrics.max_score:.2f} < {UMBRAL_RETORNO})")
                self.degradation_counter = 0
                return 'PROVIDER1', f"Retorno: max_score {metrics.max_score:.2f} < {UMBRAL_RETORNO}", metrics
            else:
                logging.info(f"⏳ Permanecer en PROVIDER2 (max_score {metrics.max_score:.2f} >= {UMBRAL_RETORNO})")
                return 'PROVIDER2', "Condiciones no mejoradas", metrics
        
        # ====================================================================
        # 3️⃣ Si estamos en PROVIDER1: verificar degradación
        # ====================================================================
        if metrics.max_score > UMBRAL_FAILOVER:
            self.degradation_counter += 1
            logging.info(
                f"⏱️ Degradación sostenida: {self.degradation_counter}/{SUSTAINED_DEGRADATION_CYCLES} ciclos "
                f"[max_score {metrics.max_score:.2f} > {UMBRAL_FAILOVER}]"
            )
            
            if self.degradation_counter >= SUSTAINED_DEGRADATION_CYCLES:
                logging.info(f"🔄 FAILOVER: {self.current_primary_provider} → PROVIDER2")
                self.degradation_counter = 0
                return 'PROVIDER2', f"Degradación sostenida: max_score {metrics.max_score:.2f} > {UMBRAL_FAILOVER}", metrics
            
            return self.current_primary_provider, f"Degradación en progreso: {self.degradation_counter}/{SUSTAINED_DEGRADATION_CYCLES}", metrics
        else:
            # Score normal, resetear contador
            if self.degradation_counter > 0:
                logging.info(f"✅ Score normalizado ({metrics.max_score:.2f} <= {UMBRAL_FAILOVER}), contador reseteado")
            self.degradation_counter = 0
            return self.current_primary_provider, "Condiciones estables", metrics
    
    def switch_to_provider(self, new_provider: str, reason: str):
        """Cambia al nuevo provider"""
        if new_provider == self.current_primary_provider:
            return
        
        logging.info(f"🔄 Cambiando de {self.current_primary_provider} a {new_provider}")
        self.current_primary_provider = new_provider
    
    def send_metrics_to_timescaledb(self, cycle_data: Dict[str, Any], metrics: LatencyMetrics):
        """
        ✅ Envía métricas a TimescaleDB usando nueva estructura bgp_metrics_new
        """
        if not self.ts_client:
            return
        
        try:
            timestamp = datetime.now(timezone.utc)
            
            metric = {
                'time': timestamp,
                'provider': 'PROVIDER1',
                'peer_ip': self.provider_peer_ip_map.get('PROVIDER1', ''),
                'peer_asn': self.provider_asn_map.get('PROVIDER1'),
                'cycle_number': self.cycle_count,
                'host': 'core-router-huawei',
                
                # Métricas del peer
                'peer_latency_ms': round(metrics.peer_avg, 2),
                'peer_jitter_ms': round(metrics.peer_stddev, 2),
                'peer_loss_pct': round(metrics.peer_loss, 2),
                
                # Métricas DNS1
                'dns1_latency_ms': round(metrics.dns1_avg, 2),
                'dns1_jitter_ms': round(metrics.dns1_stddev, 2),
                'dns1_loss_pct': round(metrics.dns1_loss, 2),
                
                # Métricas DNS2
                'dns2_latency_ms': round(metrics.dns2_avg, 2),
                'dns2_jitter_ms': round(metrics.dns2_stddev, 2),
                'dns2_loss_pct': round(metrics.dns2_loss, 2),
                
                # Scores
                'score': round(metrics.max_score, 2),
                'score_dns1': round(metrics.score_dns1, 2),
                'score_dns2': round(metrics.score_dns2, 2),
                'max_score': round(metrics.max_score, 2),
                
                # Umbrales configurables
                'umbral_failover': UMBRAL_FAILOVER,
                'umbral_retorno': UMBRAL_RETORNO,
                
                # Estado del failover
                'current_provider': cycle_data['current_provider'],
                'provider_changed': cycle_data['provider_changed'],
                'provider_change_reason': cycle_data.get('change_reason', ''),
                'degradation_cycle': self.degradation_counter,
                'sustained_degradation': self.degradation_counter >= SUSTAINED_DEGRADATION_CYCLES,
                'quality_status': self._determine_quality_status(metrics),
                'decision': cycle_data.get('decision', 'normal')
            }
            
            # Insertar en bgp_metrics_new
            self.ts_client.insert_bgp_metrics_new(metric)
            
            # Registrar evento de failover si aplica
            if cycle_data['provider_changed']:
                event = {
                    'previous_provider': cycle_data['previous_provider'],
                    'new_provider': cycle_data['new_provider'],
                    'change_reason': cycle_data['change_reason'],
                    'previous_provider_score': 0.0,
                    'new_provider_score': round(metrics.max_score, 2),
                    'detection_cycles': cycle_data['cycle'],
                    'detected_by': 'bgp_failover_engine'
                }
                result = self.ts_client.insert_failover_event(event)
                if result:
                    logging.info(f"✅ TimescaleDB: Failover registrado en ciclo #{cycle_data['cycle']}")
        
        except Exception as e:
            logging.error(f"❌ Error enviando a TimescaleDB: {e}")
    
    def _determine_quality_status(self, metrics: LatencyMetrics) -> str:
        """Determina el estado de calidad basado en métricas"""
        if (
            metrics.peer_loss >= IMMEDIATE_FAILOVER_PACKET_LOSS or
            metrics.dns1_loss >= IMMEDIATE_FAILOVER_PACKET_LOSS or
            metrics.dns2_loss >= IMMEDIATE_FAILOVER_PACKET_LOSS or
            metrics.has_peer_critical or
            metrics.has_dns1_critical or
            metrics.has_dns2_critical
        ):
            return "critical"
        
        if (
            metrics.has_peer_warning or
            metrics.has_dns1_warning or
            metrics.has_dns2_warning or
            metrics.has_packet_loss
        ):
            return "warning"
        
        return "excellent"
    
    def run_cycle(self):
        """Ejecuta un ciclo completo de monitoreo"""
        try:
            logging.info("=" * 80)
            logging.info(f"🔍 Ciclo #{self.cycle_count} - Primary: {self.current_primary_provider}")
            
            new_provider, reason, metrics = self.should_switch_provider()
            provider_will_change = new_provider != self.current_primary_provider
            
            cycle_data = {
                "cycle": self.cycle_count,
                "current_provider": self.current_primary_provider,
                "provider_changed": provider_will_change,
                "previous_provider": self.current_primary_provider if provider_will_change else None,
                "new_provider": new_provider if provider_will_change else None,
                "change_reason": reason,
                "decision": self._determine_decision(metrics, provider_will_change)
            }
            
            # Enviar métricas a TimescaleDB
            if metrics:
                self.send_metrics_to_timescaledb(cycle_data, metrics)
            
            # Ejecutar failover si aplica
            if provider_will_change:
                self.switch_to_provider(new_provider, reason)
                logging.info(f"🔄 Failover: {cycle_data['previous_provider']} → {new_provider} (ciclo: {self.cycle_count})")
                self.degradation_counter = 0
            else:
                logging.info(f"✓ Sin cambios - {reason}")
            
            self.cycle_count += 1
        
        except Exception as e:
            logging.error(f"❌ Error en ciclo: {e}", exc_info=True)
    
    def _determine_decision(self, metrics: Optional[LatencyMetrics], provider_will_change: bool) -> str:
        """Determina el tipo de decisión tomada"""
        if not metrics:
            return "error"
        
        if provider_will_change:
            if self.current_primary_provider == 'PROVIDER2' and metrics.max_score < UMBRAL_RETORNO:
                return "retorno"
            elif not metrics.is_healthy:
                return "failover_inmediato"
            else:
                return "failover"
        
        if metrics.max_score > UMBRAL_FAILOVER and self.degradation_counter > 0:
            return "degradacion"
        
        return "normal"


def main():
    """Función principal"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/var/log/bgp_failover.log')
        ]
    )
    
    try:
        subprocess.run(['mtr', '--version'], capture_output=True, check=True)
    except:
        logging.error("❌ MTR no instalado. Instalar con: apt-get install mtr")
        return 1
    
    engine = BGPFailoverEngine()
    
    logging.info("🚀 BGP Failover Engine - Versión DNS1/DNS2 con MTR Doble y Fallback")
    logging.info(f"📍 Monitoreo: PROVIDER1 (peer + DNS1 + DNS2)")
    logging.info(f"⏱️ Ciclo: {CYCLE_INTERVAL}s")
    logging.info(f"🎯 Umbral Failover: {UMBRAL_FAILOVER}")
    logging.info(f"🎯 Umbral Retorno: {UMBRAL_RETORNO}")
    logging.info(f"📊 DNS1 ({DNS_DESTINATIONS['DNS1']}): "
                f"warning={DNS_THRESHOLDS['DNS1']['warning']}ms, "
                f"critical={DNS_THRESHOLDS['DNS1']['critical']}ms")
    logging.info(f"📊 DNS2 ({DNS_DESTINATIONS['DNS2']}): "
                f"warning={DNS_THRESHOLDS['DNS2']['warning']}ms, "
                f"critical={DNS_THRESHOLDS['DNS2']['critical']}ms")
    logging.info("=" * 80)
    
    while True:
        engine.run_cycle()
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    exit(main())
