#!/usr/bin/env python3
"""
BGP Failover Engine - v2.1: SCORING NORMALIZADO (RATIO-A-UMBRAL + CAP + SEVERIDAD)
✅ LÓGICA DE SCORING (reemplaza el modelo legacy ponderado sin normalizar):
├─ Normalización ratio-a-umbral por métrica: metric_norm = min(valor/umbral, cap)
├─ Cap = 3.0 (límite de influencia, no de medición)
├─ base_score = peer_norm*0.27 + dns_norm*0.50 + jitter_norm*0.23  (pesos suman 1)
├─ loss_norm agregado sobre ventana de confirmación (3 ciclos = 15 paquetes)
├─ jitter_norm TAMBIÉN agregado en ventana (3 ciclos) — v2.1: con n=5 paquetes
│   por ciclo el StDev es muy volátil; un solo pico dominaba el score (visto
│   en corrida real: ciclo 3 en Containerlab). Windowing + peso reducido
│   corrige esto.
├─ severity_multiplier = 1 + loss_norm  →  score_final = base_score * severity_multiplier
├─ Score independiente por DNS1 y DNS2, decisión usa max(score_dns1, score_dns2)
├─ umbral_failover = 1.10 | umbral_retorno = 0.80 (constantes universales,
│   válidas porque el score está normalizado: 1.0 = justo en el límite crítico)
├─ 3 ciclos de degradación sostenida (anti-flapping)
└─ Bypass de seguridad: pérdida severa en UN solo ciclo → failover inmediato

✅ MONITOREO:
├─ Solo PROVIDER1 (peer + DNS1 + DNS2)
├─ MTR se ejecuta DOS veces por ciclo (DNS1 y DNS2) con fallback de peer
└─ Nueva tabla bgp_metrics_new (requiere migration_v2_scoring.sql aplicada)

⚠️ DECISIÓN DE DISEÑO EXPLÍCITA (heredada de la definición de fórmula acordada):
El jitter y la pérdida usados en el score son los del PATH a cada DNS
(dns1_stddev/dns2_stddev, dns1_loss/dns2_loss) — NO se promedian con el
jitter/pérdida del propio peer BGP. peer_avg es la única señal del peer que
entra al score. peer_jitter_ms y peer_loss_pct se siguen registrando en bruto
para que Etapa 3 (XGBoost) pueda evaluar empíricamente si deberían aportar.
"""
import time
import logging
import subprocess
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
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

# ⚠️ Bypass de seguridad: pérdida en UN solo ciclo (no ventana) que dispara
# failover inmediato sin esperar los 3 ciclos de confirmación. Se mantiene el
# valor heredado del motor legacy (20.0%); queda abierto a revisión porque con
# el nuevo cap+multiplicador de severidad, la ruta "graduada" ya reacciona
# rápido a pérdida sostenida — este bypass es solo para el caso de corte total
# o casi total en un único ciclo.
IMMEDIATE_FAILOVER_LOSS_PCT = 20.0

# ✅ NORMALIZACIÓN — Etapa 1 (ratio-a-umbral)
CAP = 3.0                    # límite de influencia de cualquier métrica normalizada
LOSS_SLA_PCT = 1.0            # SLA contractual de pérdida de paquetes
LOSS_WINDOW_CYCLES = 3        # ventana de confirmación para agregar pérdida (15 paquetes)
JITTER_WINDOW_CYCLES = 3      # ventana para agregar jitter y amortiguar picos de un solo ciclo (n=5)

# ✅ UMBRALES CRÍTICOS (denominadores de la normalización)
PEER_THRESHOLDS = {
    'warning': 5.0,
    'critical': 10.0,   # umbral_critico usado en peer_norm = min(peer_avg/10, cap)
}

DNS_DESTINATIONS = {
    'DNS1': '2001:db8:8888::100',
    'DNS2': '2001:db8:4444::100',
}

DNS_THRESHOLDS = {
    'DNS1': {'warning': 15.0, 'critical': 30.0},   # dns1_norm = min(dns1_avg/30, cap)
    'DNS2': {'warning': 30.0, 'critical': 60.0},   # dns2_norm = min(dns2_avg/60, cap)
}

JITTER_THRESHOLDS = {
    'warning': 5.0,
    'critical': 10.0,   # jitterX_norm = min(dnsX_stddev/10, cap)
}

# ✅ PESOS DE SCORING v2.1 — dns_norm aumentado, jitter_norm reducido.
# Motivo (evidencia empírica, ciclos 3-5 de la corrida en Containerlab):
# con n=5 paquetes/ciclo, jitter (StDev) es muy volátil — un solo pico puede
# tocar el cap (3.0) y dominar el score aunque la latencia real (señal más
# estable) sea la que efectivamente indica degradación sostenida. Bajar el
# peso de jitter, junto con agregarlo en ventana (ver JITTER_WINDOW_CYCLES),
# evita que un pico de un solo ciclo domine o "enmascare" la tendencia real.
SCORING_WEIGHTS = {
    'peer': 0.27,
    'dns': 0.50,
    'jitter': 0.23,
}

# ✅ UMBRALES DE DECISIÓN — constantes universales porque el score está
# normalizado (1.0 = justo en el límite crítico ponderado)
UMBRAL_FAILOVER = 1.10   # 10% de margen sobre el límite crítico teórico
UMBRAL_RETORNO = 0.80    # banda de histéresis del 20% por debajo del límite

ROLLING_HISTORY_SIZE = 10


@dataclass
class LatencyMetrics:
    """Métricas de latencia para PROVIDER1 con DNS1 y DNS2 separados,
    más los componentes normalizados de la fórmula de scoring v2."""

    # --- Métricas crudas ---
    peer_avg: float
    peer_loss: float
    peer_stddev: float

    dns1_avg: float
    dns1_loss: float
    dns1_stddev: float

    dns2_avg: float
    dns2_loss: float
    dns2_stddev: float

    # --- Componentes normalizados (se completan en calculate_scores) ---
    peer_norm: float = 0.0
    dns1_norm: float = 0.0
    dns2_norm: float = 0.0
    jitter1_norm: float = 0.0
    jitter2_norm: float = 0.0

    jitter1_window_ms: float = 0.0
    jitter2_window_ms: float = 0.0

    loss1_window_pct: float = 0.0
    loss2_window_pct: float = 0.0
    loss1_norm: float = 0.0
    loss2_norm: float = 0.0

    base_score_dns1: float = 0.0
    base_score_dns2: float = 0.0
    severity_multiplier_dns1: float = 1.0
    severity_multiplier_dns2: float = 1.0

    score_dns1: float = 0.0
    score_dns2: float = 0.0
    max_score: float = 0.0

    scores_ready: bool = False   # evita usar scores antes de calculate_scores()

    @property
    def is_healthy(self) -> bool:
        """Bypass de seguridad: pérdida severa en un solo ciclo (sin ventana)."""
        return (
            self.peer_loss < IMMEDIATE_FAILOVER_LOSS_PCT and
            self.dns1_loss < IMMEDIATE_FAILOVER_LOSS_PCT and
            self.dns2_loss < IMMEDIATE_FAILOVER_LOSS_PCT
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
    def has_jitter1_critical(self) -> bool:
        return self.dns1_stddev >= JITTER_THRESHOLDS['critical']

    @property
    def has_jitter2_critical(self) -> bool:
        return self.dns2_stddev >= JITTER_THRESHOLDS['critical']

    @property
    def has_packet_loss(self) -> bool:
        return self.peer_loss > 0.0 or self.dns1_loss > 0.0 or self.dns2_loss > 0.0

    def calculate_scores(
        self,
        loss1_window_pct: float,
        loss2_window_pct: float,
        jitter1_window_ms: float,
        jitter2_window_ms: float,
    ):
        """
        ✅ Fórmula de scoring v2.1 (ratio-a-umbral + cap + severidad por pérdida
        + jitter agregado en ventana).

        1. Normalización:      metric_norm = min(valor / umbral_critico, cap)
        2. Score base:         base_score = peer_norm*0.27 + dns_norm*0.50 + jitter_norm*0.23
        3. Severidad:          severity_multiplier = 1 + loss_norm
        4. Score final:        score = base_score * severity_multiplier

        loss1/2_window_pct y jitter1/2_window_ms deben venir ya agregados sobre
        la ventana de confirmación (LOSS_WINDOW_CYCLES / JITTER_WINDOW_CYCLES) —
        ver BGPFailoverEngine._compute_loss_windows() / _compute_jitter_windows().

        ⚠️ El jitter se agrega en ventana (no crudo de un solo ciclo) porque con
        n=5 paquetes por MTR, el StDev es muy volátil: un solo paquete con
        retardo anómalo puede tocar el cap y dominar el score, enmascarando
        tanto falsos positivos (spike aislado) como falsos negativos (score
        cae cuando el spike pasa, aunque la latencia siga degradada).
        """
        self.loss1_window_pct = loss1_window_pct
        self.loss2_window_pct = loss2_window_pct
        self.jitter1_window_ms = jitter1_window_ms
        self.jitter2_window_ms = jitter2_window_ms

        # --- 1. Normalización ratio-a-umbral ---
        self.peer_norm = min(self.peer_avg / PEER_THRESHOLDS['critical'], CAP)
        self.dns1_norm = min(self.dns1_avg / DNS_THRESHOLDS['DNS1']['critical'], CAP)
        self.dns2_norm = min(self.dns2_avg / DNS_THRESHOLDS['DNS2']['critical'], CAP)
        self.jitter1_norm = min(jitter1_window_ms / JITTER_THRESHOLDS['critical'], CAP)
        self.jitter2_norm = min(jitter2_window_ms / JITTER_THRESHOLDS['critical'], CAP)
        self.loss1_norm = min(loss1_window_pct / LOSS_SLA_PCT, CAP)
        self.loss2_norm = min(loss2_window_pct / LOSS_SLA_PCT, CAP)

        # --- 2. Score base (señales continuas, pesos suman 1) ---
        self.base_score_dns1 = (
            self.peer_norm * SCORING_WEIGHTS['peer'] +
            self.dns1_norm * SCORING_WEIGHTS['dns'] +
            self.jitter1_norm * SCORING_WEIGHTS['jitter']
        )
        self.base_score_dns2 = (
            self.peer_norm * SCORING_WEIGHTS['peer'] +
            self.dns2_norm * SCORING_WEIGHTS['dns'] +
            self.jitter2_norm * SCORING_WEIGHTS['jitter']
        )

        # --- 3. Multiplicador de severidad por pérdida (ventana) ---
        self.severity_multiplier_dns1 = 1 + self.loss1_norm
        self.severity_multiplier_dns2 = 1 + self.loss2_norm

        # --- 4. Score final por punto de medición ---
        self.score_dns1 = self.base_score_dns1 * self.severity_multiplier_dns1
        self.score_dns2 = self.base_score_dns2 * self.severity_multiplier_dns2
        self.max_score = max(self.score_dns1, self.score_dns2)

        self.scores_ready = True


class BGPFailoverEngine:
    """Motor de failover BGP con monitoreo DNS1/DNS2 (MTR doble con fallback)
    y scoring normalizado v2."""

    def __init__(self):
        self.ts_client = None
        self.provider_asn_map = {}
        self.provider_peer_ip_map = {}

        # Historial para rolling stats y ventana de pérdida
        self.metrics_history: List[LatencyMetrics] = []

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

        logging.info("🚀 Engine inicializado (scoring v2 - ratio-a-umbral):")
        logging.info(f"   Ciclo actual: {self.cycle_count}")
        logging.info(f"   Provider actual: {self.current_primary_provider}")
        logging.info(f"   🎯 Umbral Failover: {UMBRAL_FAILOVER} | Umbral Retorno: {UMBRAL_RETORNO}")
        logging.info(f"   📏 Cap: {CAP} | SLA pérdida: {LOSS_SLA_PCT}% | Ventana pérdida: {LOSS_WINDOW_CYCLES} ciclos")
        logging.info(f"   📊 Peer critical: {PEER_THRESHOLDS['critical']}ms")
        logging.info(f"   📊 DNS1 critical: {DNS_THRESHOLDS['DNS1']['critical']}ms | "
                     f"DNS2 critical: {DNS_THRESHOLDS['DNS2']['critical']}ms")
        logging.info(f"   📊 Jitter critical: {JITTER_THRESHOLDS['critical']}ms")
        logging.info(f"   📊 Pesos: {SCORING_WEIGHTS}")

    # ------------------------------------------------------------------
    # Carga de configuración / estado previo desde TimescaleDB
    # ------------------------------------------------------------------
    def _load_provider_config(self):
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
            logging.info("ℹ️ No hay eventos previos, usando PROVIDER1")
            return "PROVIDER1"
        except Exception as e:
            logging.error(f"⚠️ Error leyendo provider actual: {e}")
            return "PROVIDER1"

    # ------------------------------------------------------------------
    # Medición (MTR)
    # ------------------------------------------------------------------
    def run_mtr(self, destination: str, ip_version: str) -> Optional[Dict[str, Any]]:
        try:
            cmd = [
                'mtr', f'-{ip_version}', '-n', '-j',
                '-c', str(MTR_CONFIG['count']),
                '-s', str(MTR_CONFIG['packet_size']),
                '-i', str(MTR_CONFIG['interval']),
                destination
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=MTR_CONFIG['timeout']
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
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
        """Extrae peer + un DNS específico. Fallback de peer si no aparece en el hop."""
        try:
            hubs = mtr_report['report']['hubs']
            peer_ip = PEER_IPS['PROVIDER1']
            dns_ip = DNS_DESTINATIONS[dns_name]

            peer_hop = None
            dns_hop = None
            for hop in hubs:
                host = hop.get('host')
                if host == peer_ip:
                    peer_hop = hop
                elif host == dns_ip:
                    dns_hop = hop

            if not peer_hop:
                # 🔎 DIAGNÓSTICO: volcar los hops reales para identificar por qué
                # el peer configurado (peer_ip) nunca aparece en la traza a este
                # DNS. Útil para descartar ECMP / next-hop distinto / formato de
                # IP no coincidente. Quitar o bajar a DEBUG una vez diagnosticado.
                observed_hosts = [hop.get('host') for hop in hubs]
                logging.warning(
                    f"🔎 DIAGNÓSTICO: peer_ip esperado='{peer_ip}' NO está en hops de {dns_name}. "
                    f"Hops observados: {observed_hosts}"
                )
                if peer_fallback:
                    logging.info(f"ℹ️ Peer no encontrado en MTR a {dns_name}, usando fallback de DNS1")
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
        Ejecuta MTR dos veces (DNS1, DNS2) con fallback de peer y devuelve
        un LatencyMetrics con métricas CRUDAS únicamente. Los scores se
        calculan después, en should_switch_provider(), porque requieren la
        ventana de pérdida (que depende del historial ya actualizado).
        """
        ip_version = IP_VERSIONS.get('PROVIDER1', '6')

        logging.info(f"📡 Ejecutando MTR a DNS1 ({DNS_DESTINATIONS['DNS1']})...")
        mtr_dns1 = self.run_mtr(DNS_DESTINATIONS['DNS1'], ip_version)
        if not mtr_dns1:
            logging.error("❌ No se pudo obtener MTR a DNS1")
            return None

        metrics_dns1 = self.extract_metrics_single_dns(mtr_dns1, 'DNS1')
        if not metrics_dns1:
            logging.error("❌ No se pudieron extraer métricas de DNS1")
            return None

        peer_from_dns1 = {
            'peer_avg': metrics_dns1['peer_avg'],
            'peer_loss': metrics_dns1['peer_loss'],
            'peer_stddev': metrics_dns1['peer_stddev']
        }

        logging.info(f"📡 Ejecutando MTR a DNS2 ({DNS_DESTINATIONS['DNS2']})...")
        mtr_dns2 = self.run_mtr(DNS_DESTINATIONS['DNS2'], ip_version)
        if not mtr_dns2:
            logging.error("❌ No se pudo obtener MTR a DNS2")
            return None

        metrics_dns2 = self.extract_metrics_single_dns(mtr_dns2, 'DNS2', peer_fallback=peer_from_dns1)
        if not metrics_dns2:
            logging.error("❌ No se pudieron extraer métricas de DNS2")
            return None

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

        logging.info(
            f"📊 PROVIDER1 (crudo) - Peer: {combined_metrics.peer_avg:.2f}ms | "
            f"DNS1: {combined_metrics.dns1_avg:.2f}ms (loss {combined_metrics.dns1_loss:.1f}%) | "
            f"DNS2: {combined_metrics.dns2_avg:.2f}ms (loss {combined_metrics.dns2_loss:.1f}%)"
        )
        return combined_metrics

    # ------------------------------------------------------------------
    # Ventana de pérdida y rolling stats
    # ------------------------------------------------------------------
    def _compute_loss_windows(self) -> Tuple[float, float]:
        """
        Promedia dns1_loss/dns2_loss de los últimos LOSS_WINDOW_CYCLES ciclos
        (incluyendo el actual, ya en self.metrics_history). Como cada ciclo
        envía la misma cantidad de paquetes, el promedio simple equivale a
        paquetes_perdidos / paquetes_totales_en_la_ventana.
        """
        window = self.metrics_history[-LOSS_WINDOW_CYCLES:]
        loss1_window_pct = float(np.mean([m.dns1_loss for m in window]))
        loss2_window_pct = float(np.mean([m.dns2_loss for m in window]))
        return loss1_window_pct, loss2_window_pct

    def _compute_jitter_windows(self) -> Tuple[float, float]:
        """
        Promedia dns1_stddev/dns2_stddev de los últimos JITTER_WINDOW_CYCLES
        ciclos (incluyendo el actual). Amortigua picos de un solo ciclo
        producidos por el bajo tamaño de muestra de MTR (n=5 paquetes).
        """
        window = self.metrics_history[-JITTER_WINDOW_CYCLES:]
        jitter1_window_ms = float(np.mean([m.dns1_stddev for m in window]))
        jitter2_window_ms = float(np.mean([m.dns2_stddev for m in window]))
        return jitter1_window_ms, jitter2_window_ms

    def calculate_rolling_stats(self, field_name: str = 'max_score') -> Dict[str, Optional[float]]:
        """Rolling mean/std/p95 sobre un campo numérico del historial
        (default: max_score, usado para rolling_mean/std/p95 en TimescaleDB)."""
        values = [
            getattr(m, field_name) for m in self.metrics_history
            if getattr(m, field_name, None) is not None
        ]
        if len(values) < 3:
            return {'mean': None, 'std': None, 'p95': None, 'count': len(values)}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'p95': float(np.percentile(values, 95)),
            'count': len(values)
        }

    def _compute_peer_zscore(self) -> Tuple[Optional[float], Optional[str]]:
        """
        Z-score rolling de peer_avg específicamente (columna z_score_peer).
        z_score_severity es un bucket categórico derivado, no un número:
        coherente con el tipo varchar(20) definido en el schema.
        """
        peer_values = [m.peer_avg for m in self.metrics_history]
        if len(peer_values) < 3:
            return None, None

        mean = float(np.mean(peer_values))
        std = float(np.std(peer_values))
        if std == 0:
            return 0.0, 'normal'

        current = peer_values[-1]
        z = (current - mean) / std
        abs_z = abs(z)
        if abs_z < 1.0:
            bucket = 'normal'
        elif abs_z < 2.0:
            bucket = 'elevated'
        else:
            bucket = 'severe'
        return float(z), bucket

    # ------------------------------------------------------------------
    # Lógica de decisión
    # ------------------------------------------------------------------
    def should_switch_provider(self) -> Tuple[str, str, Optional[LatencyMetrics], bool]:
        """
        1. Medir PROVIDER1 (peer + DNS1 + DNS2) → métricas crudas
        2. Agregar al historial y calcular ventana de pérdida
        3. Calcular scores normalizados (v2) sobre ese historial
        4. Bypass de seguridad: pérdida severa en un ciclo → failover inmediato
        5. Si estamos en PROVIDER2: evaluar retorno (max_score < umbral_retorno)
        6. Si estamos en PROVIDER1: evaluar degradación sostenida (3 ciclos)

        Devuelve además un booleano `immediate_trigger` para distinguir en la
        BD el bypass de seguridad de una decisión graduada por score.
        """
        metrics = self.measure_provider_latency()
        if not metrics:
            logging.error("❌ No se pudieron obtener métricas")
            return self.current_primary_provider, "Error de medición", None, False

        # Historial + ventana de pérdida (debe ir antes de calcular scores)
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > ROLLING_HISTORY_SIZE:
            self.metrics_history.pop(0)

        loss1_window_pct, loss2_window_pct = self._compute_loss_windows()
        jitter1_window_ms, jitter2_window_ms = self._compute_jitter_windows()
        metrics.calculate_scores(
            loss1_window_pct, loss2_window_pct, jitter1_window_ms, jitter2_window_ms
        )

        logging.info("📈 Estado actual:")
        logging.info(f"   ⭐ Provider actual: {self.current_primary_provider}")
        logging.info(
            f"   📊 Score DNS1: {metrics.score_dns1:.3f} (base {metrics.base_score_dns1:.3f} × "
            f"sev {metrics.severity_multiplier_dns1:.2f}) | "
            f"Score DNS2: {metrics.score_dns2:.3f} (base {metrics.base_score_dns2:.3f} × "
            f"sev {metrics.severity_multiplier_dns2:.2f}) | Max: {metrics.max_score:.3f}"
        )
        logging.info(f"   🎯 Umbral Failover: {UMBRAL_FAILOVER} | Umbral Retorno: {UMBRAL_RETORNO}")

        # 1️⃣ Bypass de seguridad — pérdida severa en un solo ciclo
        if not metrics.is_healthy:
            worst_loss = max(metrics.peer_loss, metrics.dns1_loss, metrics.dns2_loss)
            logging.warning(f"⚠️ Pérdida crítica en un ciclo: {worst_loss:.1f}% — bypass de seguridad")
            self.degradation_counter = 0
            new_provider = 'PROVIDER2' if self.current_primary_provider == 'PROVIDER1' else 'PROVIDER1'
            return new_provider, f"Bypass de seguridad: pérdida {worst_loss:.1f}% en un ciclo", metrics, True

        # 2️⃣ En PROVIDER2: evaluar retorno
        if self.current_primary_provider == 'PROVIDER2':
            if metrics.max_score < UMBRAL_RETORNO:
                logging.info(f"✅ Retorno a PROVIDER1 (max_score {metrics.max_score:.3f} < {UMBRAL_RETORNO})")
                self.degradation_counter = 0
                return 'PROVIDER1', f"Retorno: max_score {metrics.max_score:.3f} < {UMBRAL_RETORNO}", metrics, False
            logging.info(f"⏳ Permanecer en PROVIDER2 (max_score {metrics.max_score:.3f} >= {UMBRAL_RETORNO})")
            return 'PROVIDER2', "Condiciones no mejoradas", metrics, False

        # 3️⃣ En PROVIDER1: evaluar degradación sostenida
        if metrics.max_score > UMBRAL_FAILOVER:
            self.degradation_counter += 1
            logging.info(
                f"⏱️ Degradación sostenida: {self.degradation_counter}/{SUSTAINED_DEGRADATION_CYCLES} ciclos "
                f"[max_score {metrics.max_score:.3f} > {UMBRAL_FAILOVER}]"
            )
            if self.degradation_counter >= SUSTAINED_DEGRADATION_CYCLES:
                logging.info(f"🔄 FAILOVER: {self.current_primary_provider} → PROVIDER2")
                self.degradation_counter = 0
                return 'PROVIDER2', f"Degradación sostenida: max_score {metrics.max_score:.3f} > {UMBRAL_FAILOVER}", metrics, False
            return self.current_primary_provider, f"Degradación en progreso: {self.degradation_counter}/{SUSTAINED_DEGRADATION_CYCLES}", metrics, False

        if self.degradation_counter > 0:
            logging.info(f"✅ Score normalizado ({metrics.max_score:.3f} <= {UMBRAL_FAILOVER}), contador reseteado")
        self.degradation_counter = 0
        return self.current_primary_provider, "Condiciones estables", metrics, False

    def switch_to_provider(self, new_provider: str, reason: str):
        if new_provider == self.current_primary_provider:
            return
        logging.info(f"🔄 Cambiando de {self.current_primary_provider} a {new_provider}")
        self.current_primary_provider = new_provider

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def send_metrics_to_timescaledb(
        self, cycle_data: Dict[str, Any], metrics: LatencyMetrics, immediate_trigger: bool
    ):
        if not self.ts_client:
            return
        try:
            timestamp = datetime.now(timezone.utc)
            rolling = self.calculate_rolling_stats('max_score')
            z_peer, z_peer_bucket = self._compute_peer_zscore()

            metric = {
                'time': timestamp,
                'provider': 'PROVIDER1',
                'peer_ip': self.provider_peer_ip_map.get('PROVIDER1', ''),
                'peer_asn': self.provider_asn_map.get('PROVIDER1'),
                'cycle_number': self.cycle_count,
                'host': 'core-router-huawei',

                # Métricas crudas
                'peer_latency_ms': round(metrics.peer_avg, 2),
                'peer_jitter_ms': round(metrics.peer_stddev, 2),
                'peer_loss_pct': round(metrics.peer_loss, 2),
                'dns1_latency_ms': round(metrics.dns1_avg, 2),
                'dns1_jitter_ms': round(metrics.dns1_stddev, 2),
                'dns1_loss_pct': round(metrics.dns1_loss, 2),
                'dns2_latency_ms': round(metrics.dns2_avg, 2),
                'dns2_jitter_ms': round(metrics.dns2_stddev, 2),
                'dns2_loss_pct': round(metrics.dns2_loss, 2),

                # Componentes normalizados (Etapa 1 - v2)
                'peer_norm': round(metrics.peer_norm, 4),
                'dns1_norm': round(metrics.dns1_norm, 4),
                'dns2_norm': round(metrics.dns2_norm, 4),
                'jitter1_norm': round(metrics.jitter1_norm, 4),
                'jitter2_norm': round(metrics.jitter2_norm, 4),
                'jitter1_window_ms': round(metrics.jitter1_window_ms, 3),
                'jitter2_window_ms': round(metrics.jitter2_window_ms, 3),
                'loss1_window_pct': round(metrics.loss1_window_pct, 3),
                'loss2_window_pct': round(metrics.loss2_window_pct, 3),
                'loss1_norm': round(metrics.loss1_norm, 4),
                'loss2_norm': round(metrics.loss2_norm, 4),
                'base_score_dns1': round(metrics.base_score_dns1, 4),
                'base_score_dns2': round(metrics.base_score_dns2, 4),
                'severity_multiplier_dns1': round(metrics.severity_multiplier_dns1, 4),
                'severity_multiplier_dns2': round(metrics.severity_multiplier_dns2, 4),

                # Hiperparámetros vigentes (reproducibilidad del dataset ML)
                'weight_peer': SCORING_WEIGHTS['peer'],
                'weight_dns': SCORING_WEIGHTS['dns'],
                'weight_jitter': SCORING_WEIGHTS['jitter'],
                'cap_value': CAP,
                'loss_sla_pct': LOSS_SLA_PCT,

                # Scores finales
                'score': round(metrics.max_score, 4),           # alias legacy de max_score
                'score_dns1': round(metrics.score_dns1, 4),
                'score_dns2': round(metrics.score_dns2, 4),
                'max_score': round(metrics.max_score, 4),

                # Umbrales de decisión vigentes
                'umbral_failover': UMBRAL_FAILOVER,
                'umbral_retorno': UMBRAL_RETORNO,

                # Estado del failover
                'current_provider': cycle_data['current_provider'],
                'provider_changed': cycle_data['provider_changed'],
                'provider_change_reason': cycle_data.get('change_reason', ''),
                'degradation_cycle': self.degradation_counter,
                'sustained_degradation': self.degradation_counter >= SUSTAINED_DEGRADATION_CYCLES,
                'quality_status': self._determine_quality_status(metrics),
                'decision': cycle_data.get('decision', 'normal'),
                'immediate_failover_triggered': immediate_trigger,

                # Rolling stats / Z-score (poblados desde ahora, insumo para Etapa 2)
                'rolling_mean': rolling['mean'],
                'rolling_std': rolling['std'],
                'rolling_p95': rolling['p95'],
                'z_score_peer': round(z_peer, 4) if z_peer is not None else None,
                'z_score_severity': z_peer_bucket,
            }

            self.ts_client.insert_bgp_metrics_new(metric)

            if cycle_data['provider_changed']:
                event = {
                    'previous_provider': cycle_data['previous_provider'],
                    'new_provider': cycle_data['new_provider'],
                    'change_reason': cycle_data['change_reason'],
                    'previous_provider_score': 0.0,
                    'new_provider_score': round(metrics.max_score, 4),
                    'detection_cycles': cycle_data['cycle'],
                    'detected_by': 'bgp_failover_engine_v2'
                }
                result = self.ts_client.insert_failover_event(event)
                if result:
                    logging.info(f"✅ TimescaleDB: Failover registrado en ciclo #{cycle_data['cycle']}")

        except Exception as e:
            logging.error(f"❌ Error enviando a TimescaleDB: {e}")

    def _determine_quality_status(self, metrics: LatencyMetrics) -> str:
        """
        Bucket de calidad ATADO a los mismos umbrales que usa la decisión de
        failover (antes usaba umbrales ms independientes del score, lo cual
        podía mostrar 'excellent' mientras el score ya disparaba failover).
        """
        if not metrics.is_healthy or metrics.max_score > UMBRAL_FAILOVER:
            return "critical"
        if metrics.max_score >= UMBRAL_RETORNO:
            return "warning"
        return "excellent"

    # ------------------------------------------------------------------
    # Ciclo principal
    # ------------------------------------------------------------------
    def run_cycle(self):
        try:
            logging.info("=" * 80)
            logging.info(f"🔍 Ciclo #{self.cycle_count} - Primary: {self.current_primary_provider}")

            new_provider, reason, metrics, immediate_trigger = self.should_switch_provider()
            provider_will_change = new_provider != self.current_primary_provider

            cycle_data = {
                "cycle": self.cycle_count,
                "current_provider": self.current_primary_provider,
                "provider_changed": provider_will_change,
                "previous_provider": self.current_primary_provider if provider_will_change else None,
                "new_provider": new_provider if provider_will_change else None,
                "change_reason": reason,
                "decision": self._determine_decision(metrics, provider_will_change, immediate_trigger)
            }

            if metrics:
                self.send_metrics_to_timescaledb(cycle_data, metrics, immediate_trigger)

            if provider_will_change:
                self.switch_to_provider(new_provider, reason)
                logging.info(f"🔄 Failover: {cycle_data['previous_provider']} → {new_provider} (ciclo: {self.cycle_count})")
                self.degradation_counter = 0
            else:
                logging.info(f"✓ Sin cambios - {reason}")

            self.cycle_count += 1

        except Exception as e:
            logging.error(f"❌ Error en ciclo: {e}", exc_info=True)

    def _determine_decision(
        self, metrics: Optional[LatencyMetrics], provider_will_change: bool, immediate_trigger: bool
    ) -> str:
        if not metrics:
            return "error"
        if immediate_trigger:
            return "failover_inmediato"
        if provider_will_change:
            if self.current_primary_provider == 'PROVIDER2' and metrics.max_score < UMBRAL_RETORNO:
                return "retorno"
            return "failover"
        if metrics.max_score > UMBRAL_FAILOVER and self.degradation_counter > 0:
            return "degradacion"
        return "normal"


def main():
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
    except Exception:
        logging.error("❌ MTR no instalado. Instalar con: apt-get install mtr")
        return 1

    engine = BGPFailoverEngine()

    logging.info("🚀 BGP Failover Engine v2 - Scoring normalizado (ratio-a-umbral)")
    logging.info(f"📍 Monitoreo: PROVIDER1 (peer + DNS1 + DNS2)")
    logging.info(f"⏱️ Ciclo: {CYCLE_INTERVAL}s")
    logging.info(f"🎯 Umbral Failover: {UMBRAL_FAILOVER} | Umbral Retorno: {UMBRAL_RETORNO}")
    logging.info(f"📏 Cap: {CAP} | Ventana pérdida: {LOSS_WINDOW_CYCLES} ciclos | SLA: {LOSS_SLA_PCT}%")
    logging.info("=" * 80)

    while True:
        engine.run_cycle()
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    exit(main())
