#!/usr/bin/env python3
"""
synthetic_data_generator.py — Genera datos sintéticos REUTILIZANDO el motor real
═══════════════════════════════════════════════════════════════════════════════
✅ DIFERENCIA DE FONDO respecto al generador anterior (data_generator.py):

El generador anterior reimplementaba su propia versión de la fórmula de scoring
(pesos legacy 0.4/0.6/0.5/0.5, sin normalizar), su propia lógica de detección
(z-score/absolute/relative/combined severity — un esquema que nunca existió en
el motor real), y escribía directamente en ml_features saltándose bgp_metrics_new
por completo. Resultado: un dataset "sintético" que en realidad no reflejaba el
comportamiento del sistema que construimos — como bien identificaste, no era
realista y además quedaba desincronizado cada vez que el motor cambiaba.

Este generador NO reimplementa nada de la lógica de decisión. En cambio:
├─ Importa BGPFailoverEngine y LatencyMetrics DIRECTAMENTE de
│   bgp_failover_engine_new.py (el motor real, v2.6)
├─ Sobrescribe ÚNICAMENTE measure_provider_latency() — el único punto que
│   normalmente ejecuta MTR — para devolver métricas sintéticas en vez de
│   medir la red real
├─ Todo lo demás (calculate_scores, should_switch_provider, degradation/
│   improvement counters, bypass de seguridad, send_metrics_to_timescaledb)
│   corre SIN MODIFICAR — exactamente el mismo código que ya está validado
│   en Containerlab
└─ Escribe en bgp_metrics_new (la tabla real), no en ml_features — para
   derivar ml_features se sigue usando feature_engine_incremental.py después,
   igual que con datos reales. Esto es justo lo que señalaste: generar
   directamente en ml_features se salta el pipeline real y es menos preciso,
   no menos complejo.

✅ ALCANCE v1 (según lo acordado):
├─ SIN pérdida de paquetes (loss1/loss2 = 0 siempre) — coherente con que el
│   bypass de seguridad y loss_norm/severity_multiplier no se ejercitan aún
├─ Escenarios calibrados contra el CSV real de 40 ciclos (topología propia):
│   baseline normal + episodios de degradación con rampa (no saltos
│   instantáneos) en DNS1 y/o DNS2, coherente con los dos episodios
│   degradación→failover→retorno observados en la captura real
└─ Timestamps sintéticos con intervalo real medido (~45s/ciclo, no los 30s
   nominales de CYCLE_INTERVAL — ver conversación: el ciclo real incluye el
   tiempo de ejecución de MTR además del sleep configurado)

USO:
    python3 synthetic_data_generator.py --cycles 300
    python3 synthetic_data_generator.py --cycles 300 --episode-probability 0.05 --seed 7
    python3 synthetic_data_generator.py --cycles 300 --csv-mirror /tmp/synthetic.csv
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import logging
import random
import datetime as real_datetime_module
from datetime import timedelta

import numpy as np

import bgp_failover_engine_new as engine_mod
from bgp_failover_engine_new import BGPFailoverEngine, LatencyMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ Intervalo real medido entre ciclos en Containerlab (ver conversación:
# CYCLE_INTERVAL=30s de sleep + ~15s de ejecución MTR ≈ 44.9s reales)
REAL_CYCLE_INTERVAL_SECONDS = 45

# ✅ Rangos calibrados contra el CSV real de 40 ciclos (topología propia)
BASELINE = {
    'peer_avg':  (5.5, 1.5),   # (media, desvío) para np.random.normal
    'peer_stddev': (3.0, 1.3),
    'dns1_avg':  (6.5, 2.0),
    'dns1_stddev': (3.0, 1.6),
    'dns2_avg':  (7.5, 3.0),
    'dns2_stddev': (3.0, 1.6),
}

# Severidad de pico durante un episodio de degradación (ms) — coherente con
# los breach reales observados (34-66ms sobre umbrales críticos de 30/60ms)
PEAK_SEVERITY_RANGE_DNS1 = (32, 60)   # umbral crítico DNS1 = 30ms
PEAK_SEVERITY_RANGE_DNS2 = (62, 90)   # umbral crítico DNS2 = 60ms
PEAK_JITTER_RANGE = (3, 12)           # ocasionalmente cruza el crítico de jitter (10ms)


class _FrozenDateTime(real_datetime_module.datetime):
    """
    Permite congelar datetime.now() dentro de bgp_failover_engine_new durante
    la generación, para que cada ciclo sintético quede timestamped con un
    instante controlado (no el reloj real de esta corrida) sin tocar el
    archivo del motor.
    """
    _fixed_now = None

    @classmethod
    def now(cls, tz=None):
        return cls._fixed_now


class ScenarioGenerator:
    """
    Máquina de estados simple que decide, ciclo a ciclo, qué métricas RAW
    emitir. Independiente de la lógica de decisión del motor (esa la maneja
    el motor real una vez recibe estos números) — este generador solo
    decide "qué está pasando en la red", no "qué debería hacer el motor".

    Estados: NORMAL -> RAMP_UP -> PLATEAU -> RAMP_DOWN -> NORMAL
    """

    def __init__(self, episode_probability=0.04, rng=None):
        self.episode_probability = episode_probability
        self.rng = rng or np.random.default_rng()
        self.state = 'NORMAL'
        self.target = None          # 'dns1' | 'dns2' | 'peer'
        self.state_cycles_left = 0
        self.peak_value = None
        self.ramp_start_value = None
        self.ramp_progress = 0
        self.ramp_total = 0

    def _sample_baseline(self, key):
        mean, std = BASELINE[key]
        return max(0.3, float(self.rng.normal(mean, std)))

    def _start_episode(self):
        self.target = self.rng.choice(['dns1', 'dns2'], p=[0.6, 0.4])
        if self.target == 'dns1':
            self.peak_value = float(self.rng.uniform(*PEAK_SEVERITY_RANGE_DNS1))
        else:
            self.peak_value = float(self.rng.uniform(*PEAK_SEVERITY_RANGE_DNS2))

        self.state = 'RAMP_UP'
        self.ramp_total = int(self.rng.integers(2, 4))     # 2-3 ciclos de rampa
        self.ramp_progress = 0
        self.ramp_start_value = self._sample_baseline(f'{self.target}_avg')
        self.state_cycles_left = self.ramp_total

    def next_metrics(self):
        """Devuelve dict con peer_avg/stddev, dns1_avg/stddev, dns2_avg/stddev."""
        m = {
            'peer_avg': self._sample_baseline('peer_avg'),
            'peer_stddev': max(0.1, self._sample_baseline('peer_stddev')),
            'dns1_avg': self._sample_baseline('dns1_avg'),
            'dns1_stddev': max(0.1, self._sample_baseline('dns1_stddev')),
            'dns2_avg': self._sample_baseline('dns2_avg'),
            'dns2_stddev': max(0.1, self._sample_baseline('dns2_stddev')),
        }

        if self.state == 'NORMAL':
            if self.rng.random() < self.episode_probability:
                self._start_episode()
                # cae directo a la rama RAMP_UP más abajo en esta misma llamada
            else:
                return m

        if self.state == 'RAMP_UP':
            self.ramp_progress += 1
            frac = self.ramp_progress / max(1, self.ramp_total)
            interp = self.ramp_start_value + (self.peak_value - self.ramp_start_value) * frac
            m[f'{self.target}_avg'] = interp
            m[f'{self.target}_stddev'] = max(0.2, self._sample_baseline(f'{self.target}_stddev') * (1 + frac))
            self.state_cycles_left -= 1
            if self.state_cycles_left <= 0:
                self.state = 'PLATEAU'
                self.state_cycles_left = int(self.rng.integers(3, 7))  # 3-6 ciclos sostenido

        elif self.state == 'PLATEAU':
            noise = float(self.rng.normal(0, self.peak_value * 0.03))
            m[f'{self.target}_avg'] = max(0.3, self.peak_value + noise)
            # jitter a veces también sube durante el plateau (no siempre)
            if self.rng.random() < 0.4:
                m[f'{self.target}_stddev'] = float(self.rng.uniform(*PEAK_JITTER_RANGE))
            else:
                m[f'{self.target}_stddev'] = max(0.2, self._sample_baseline(f'{self.target}_stddev') * 1.3)
            self.state_cycles_left -= 1
            if self.state_cycles_left <= 0:
                self.state = 'RAMP_DOWN'
                self.ramp_total = int(self.rng.integers(2, 5))
                self.ramp_progress = 0
                self.state_cycles_left = self.ramp_total

        elif self.state == 'RAMP_DOWN':
            self.ramp_progress += 1
            frac = self.ramp_progress / max(1, self.ramp_total)
            recovery_target = self._sample_baseline(f'{self.target}_avg')
            interp = self.peak_value + (recovery_target - self.peak_value) * frac
            m[f'{self.target}_avg'] = max(0.3, interp)
            self.state_cycles_left -= 1
            if self.state_cycles_left <= 0:
                self.state = 'NORMAL'
                self.target = None

        return m


class SyntheticBGPFailoverEngine(BGPFailoverEngine):
    """
    Subclase que reemplaza measure_provider_latency() por datos sintéticos.
    Todo el resto de la lógica (should_switch_provider, calculate_scores,
    send_metrics_to_timescaledb, etc.) es EXACTAMENTE la del motor real —
    no se sobrescribe nada más.
    """

    def __init__(self, scenario_generator: ScenarioGenerator):
        self.scenario_generator = scenario_generator
        super().__init__()

    def measure_provider_latency(self):
        raw = self.scenario_generator.next_metrics()
        metrics = LatencyMetrics(
            peer_avg=raw['peer_avg'], peer_loss=0.0, peer_stddev=raw['peer_stddev'],
            dns1_avg=raw['dns1_avg'], dns1_loss=0.0, dns1_stddev=raw['dns1_stddev'],
            dns2_avg=raw['dns2_avg'], dns2_loss=0.0, dns2_stddev=raw['dns2_stddev'],
        )
        logging.info(
            f"🎲 Sintético — Peer: {metrics.peer_avg:.2f}ms | "
            f"DNS1: {metrics.dns1_avg:.2f}ms (jit {metrics.dns1_stddev:.2f}) | "
            f"DNS2: {metrics.dns2_avg:.2f}ms (jit {metrics.dns2_stddev:.2f}) "
            f"[escenario: {self.scenario_generator.state}]"
        )
        return metrics


def generate(cycles, episode_probability, seed, start_time, interval_seconds, csv_mirror_path):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    scenario = ScenarioGenerator(episode_probability=episode_probability, rng=rng)
    engine = SyntheticBGPFailoverEngine(scenario)

    if not engine.ts_client:
        logger.error("❌ No se pudo conectar a TimescaleDB — abortando. "
                      "Verificar TIMESCALEDB_* en bgp_failover_engine_new.py")
        return

    # Capturar filas insertadas también en memoria, para el espejo CSV opcional
    mirror_rows = []
    if csv_mirror_path:
        original_insert = engine.ts_client.insert_bgp_metrics_new

        def _insert_and_mirror(metrics_dict):
            mirror_rows.append(dict(metrics_dict))
            return original_insert(metrics_dict)

        engine.ts_client.insert_bgp_metrics_new = _insert_and_mirror

    # Congelar datetime.now() dentro del módulo del motor, sin tocar su archivo
    engine_mod.datetime = _FrozenDateTime

    logger.info("=" * 80)
    logger.info(f"🎲 Generando {cycles} ciclos sintéticos (motor real v2.6, sin pérdida de paquetes)")
    logger.info(f"   Probabilidad de episodio por ciclo: {episode_probability:.1%}")
    logger.info(f"   Intervalo entre ciclos: {interval_seconds}s | Inicio: {start_time.isoformat()}")
    logger.info(f"   Provider inicial (continúa desde BD): {engine.current_primary_provider}")
    logger.info(f"   Cycle_number inicial (continúa desde BD): {engine.cycle_count}")
    logger.info("=" * 80)

    current_time = start_time
    for i in range(cycles):
        _FrozenDateTime._fixed_now = current_time
        engine.run_cycle()
        current_time = current_time + timedelta(seconds=interval_seconds)

        if (i + 1) % 50 == 0:
            logger.info(f"   ✓ {i + 1}/{cycles} ciclos generados")

    logger.info("=" * 80)
    logger.info(f"✅ Generación completa: {cycles} ciclos insertados en bgp_metrics_new")
    logger.info(f"   Rango de tiempo simulado: {start_time.isoformat()} → {current_time.isoformat()}")
    logger.info("=" * 80)

    if csv_mirror_path and mirror_rows:
        import pandas as pd
        pd.DataFrame(mirror_rows).to_csv(csv_mirror_path, index=False)
        logger.info(f"📄 Espejo CSV guardado en: {csv_mirror_path}")

    logger.info("\n🎯 Próximos pasos:")
    logger.info("   1. Ejecutar feature_engine_incremental.py para derivar ml_features")
    logger.info("      (igual que con datos reales — no se salta el pipeline)")
    logger.info("   2. Ejecutar train_from_ml_features.py para entrenar/validar el modelo")


def main():
    parser = argparse.ArgumentParser(
        description='Genera datos sintéticos en bgp_metrics_new reutilizando el motor real de failover'
    )
    parser.add_argument('--cycles', type=int, default=300, help='Cantidad de ciclos a generar (default: 300)')
    parser.add_argument('--episode-probability', type=float, default=0.04,
                         help='Probabilidad por ciclo de iniciar un episodio de degradación (default: 0.04)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla de reproducibilidad')
    parser.add_argument('--interval-seconds', type=int, default=REAL_CYCLE_INTERVAL_SECONDS,
                         help=f'Segundos simulados entre ciclos (default: {REAL_CYCLE_INTERVAL_SECONDS}, '
                              f'el intervalo real medido, no los 30s nominales de CYCLE_INTERVAL)')
    parser.add_argument('--days-back', type=float, default=None,
                         help='Si se especifica, el primer ciclo empieza hace N días (default: calculado '
                              'automáticamente para que el último ciclo termine "ahora")')
    parser.add_argument('--csv-mirror', type=str, default=None,
                         help='Ruta opcional para guardar además un CSV espejo de las filas insertadas')
    args = parser.parse_args()

    if args.days_back is not None:
        start_time = real_datetime_module.datetime.now(real_datetime_module.timezone.utc) - timedelta(days=args.days_back)
    else:
        total_span = timedelta(seconds=args.interval_seconds * args.cycles)
        start_time = real_datetime_module.datetime.now(real_datetime_module.timezone.utc) - total_span

    generate(
        cycles=args.cycles,
        episode_probability=args.episode_probability,
        seed=args.seed,
        start_time=start_time,
        interval_seconds=args.interval_seconds,
        csv_mirror_path=args.csv_mirror,
    )


if __name__ == '__main__':
    main()
