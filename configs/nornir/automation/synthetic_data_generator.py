#!/usr/bin/env python3
"""
synthetic_data_generator.py — v2: calibrado contra la operación real del ISP
═══════════════════════════════════════════════════════════════════════════════
⚠️ DECLARACIÓN DE ALCANCE Y LIMITACIÓN (importante para el informe/paper):

Este script NO pretende generar un dataset representativo de la red real del
ISP. Genera datos ALEATORIOS/SINTÉTICOS con el único propósito de EJERCITAR
el pipeline completo (motor → TimescaleDB → feature engine → entrenamiento)
de punta a punta, ante la imposibilidad práctica de capturar un dataset real
completo dentro del plazo de este trabajo.

La secuencia de contribuciones de este trabajo es, en orden:
  1. Demostrar que el motor de failover funciona correctamente en una
     topología Containerlab, almacena histórico, deriva features y permite
     entrenar un modelo — CON DATOS REALES capturados de esa topología
     (bgp_metrics_new con tráfico real generado en el laboratorio).
  2. Señalar que el valor real de Etapa 3 (recalibrar pesos/umbrales) requiere
     un dataset capturado de la operación real de un ISP — que este mismo
     stack (motor + TimescaleDB + feature engine) ya está en condiciones de
     recolectar, una vez desplegado en producción.
  3. Ante la imposibilidad de tener ESE dataset real dentro de este plazo,
     se propone este generador como banco de pruebas del pipeline de
     entrenamiento — con la limitación explícita de que es data aleatoria,
     no observación real de red.

✅ CALIBRACIÓN v2 — contra la entrevista al ISP (no contra el CSV genérico
de v1). Datos aportados por el ISP:
├─ Los umbrales de latencia que usan son IDÉNTICOS a los de esta simulación
│   (30ms/60ms DNS1/DNS2) — consistente, no requiere recalibrar severidad.
├─ Aplican failover/retorno tras **15 minutos** de latencia sostenida — no
│   los 90s (3 ciclos) de nuestro motor de laboratorio.
├─ Frecuencia real: **2-3 failovers por semana**, SIEMPRE en horario pico de
│   demanda — no eventos uniformemente distribuidos las 24h.
└─ ⚠️ Supuestos que NO se pudieron confirmar con el ISP (declarar como tales
   en el informe si se citan estos números):
   - Horario pico: se asume 4 horas/día (no confirmado por el ISP).
   - Cadencia de medición real del ISP: desconocida — el ISP monitorea con
     ping continuo pero no especificó el intervalo de muestreo. Se usa como
     aproximación la cadencia de nuestro propio motor (~45s/ciclo medida
     empíricamente), así que "15 minutos" se traduce a ~20 ciclos. Esta
     equivalencia es una aproximación de trabajo, NO un dato confirmado.

USO:
    # Modo laboratorio (v1, alta frecuencia, umbral de 3 ciclos — para
    # iterar rápido y probar que el pipeline funciona):
    python3 synthetic_data_generator.py --scale lab --cycles 300

    # Modo "realista" (calibrado contra el ISP — para el análisis de Etapa 3):
    python3 synthetic_data_generator.py --scale realistic --cycles 100000

    # Parámetros de calibración ajustables manualmente:
    python3 synthetic_data_generator.py --scale realistic --cycles 50000 \\
        --events-per-week 2.5 --peak-hour-start 19 --peak-hour-duration 4 \\
        --confirmation-cycles 20
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

REAL_CYCLE_INTERVAL_SECONDS = 45

BASELINE = {
    'peer_avg':  (5.5, 1.5),
    'peer_stddev': (3.0, 1.3),
    'dns1_avg':  (6.5, 2.0),
    'dns1_stddev': (3.0, 1.6),
    'dns2_avg':  (7.5, 3.0),
    'dns2_stddev': (3.0, 1.6),
}

PEAK_SEVERITY_RANGE_DNS1 = (32, 60)
PEAK_SEVERITY_RANGE_DNS2 = (62, 90)
PEAK_JITTER_RANGE = (3, 12)

SCALE_PRESETS = {
    'lab': {
        'confirmation_cycles': 3,
        'events_per_week': None,
        'episode_probability': 0.04,
        'peak_hour_only': False,
    },
    'realistic': {
        'confirmation_cycles': 20,
        'events_per_week': 2.5,
        'episode_probability': None,
        'peak_hour_only': True,
    },
}


class _FrozenDateTime(real_datetime_module.datetime):
    _fixed_now = None

    @classmethod
    def now(cls, tz=None):
        return cls._fixed_now


class ScenarioGenerator:
    """
    Máquina de estados que decide, ciclo a ciclo, qué métricas RAW emitir.

    v2 — dos cambios de calibración respecto a v1:
    1. La probabilidad de iniciar un episodio puede depender de si el ciclo
       actual cae en horario pico (peak_hour_only=True).
    2. La duración del PLATEAU escala con confirmation_cycles.
    """

    def __init__(self, episode_probability, confirmation_cycles=3,
                 peak_hour_only=False, peak_hour_start=19, peak_hour_duration=4,
                 rng=None):
        self.episode_probability = episode_probability
        self.confirmation_cycles = confirmation_cycles
        self.peak_hour_only = peak_hour_only
        self.peak_hour_start = peak_hour_start
        self.peak_hour_duration = peak_hour_duration
        self.rng = rng or np.random.default_rng()

        self.current_hour = None

        self.state = 'NORMAL'
        self.target = None
        self.state_cycles_left = 0
        self.peak_value = None
        self.ramp_start_value = None
        self.ramp_progress = 0
        self.ramp_total = 0

    def set_current_time(self, dt):
        self.current_hour = dt.hour

    def _in_peak_hour(self):
        if self.current_hour is None:
            return True
        end = (self.peak_hour_start + self.peak_hour_duration) % 24
        if self.peak_hour_start < end:
            return self.peak_hour_start <= self.current_hour < end
        else:
            return self.current_hour >= self.peak_hour_start or self.current_hour < end

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
        self.ramp_total = int(self.rng.integers(2, 4))
        self.ramp_progress = 0
        self.ramp_start_value = self._sample_baseline(f'{self.target}_avg')
        self.state_cycles_left = self.ramp_total

    def next_metrics(self):
        m = {
            'peer_avg': self._sample_baseline('peer_avg'),
            'peer_stddev': max(0.1, self._sample_baseline('peer_stddev')),
            'dns1_avg': self._sample_baseline('dns1_avg'),
            'dns1_stddev': max(0.1, self._sample_baseline('dns1_stddev')),
            'dns2_avg': self._sample_baseline('dns2_avg'),
            'dns2_stddev': max(0.1, self._sample_baseline('dns2_stddev')),
        }

        if self.state == 'NORMAL':
            can_start = (not self.peak_hour_only) or self._in_peak_hour()
            if can_start and self.rng.random() < self.episode_probability:
                self._start_episode()
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
                self.state_cycles_left = int(
                    self.rng.integers(self.confirmation_cycles + 2, self.confirmation_cycles + 15)
                )

        elif self.state == 'PLATEAU':
            noise = float(self.rng.normal(0, self.peak_value * 0.03))
            m[f'{self.target}_avg'] = max(0.3, self.peak_value + noise)
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


def compute_calibrated_probability(events_per_week, interval_seconds, peak_hour_duration):
    cycles_per_day = 86400 / interval_seconds
    peak_cycles_per_day = cycles_per_day * (peak_hour_duration / 24)
    peak_cycles_per_week = peak_cycles_per_day * 7
    return events_per_week / peak_cycles_per_week


def generate(cycles, scale, seed, start_time, interval_seconds, csv_mirror_path,
             events_per_week=None, peak_hour_start=19, peak_hour_duration=4,
             confirmation_cycles=None, episode_probability_override=None):
    preset = SCALE_PRESETS[scale]

    resolved_confirmation_cycles = confirmation_cycles or preset['confirmation_cycles']
    resolved_events_per_week = events_per_week if events_per_week is not None else preset['events_per_week']
    peak_hour_only = preset['peak_hour_only']

    if episode_probability_override is not None:
        resolved_probability = episode_probability_override
    elif resolved_events_per_week is not None:
        resolved_probability = compute_calibrated_probability(
            resolved_events_per_week, interval_seconds, peak_hour_duration
        )
    else:
        resolved_probability = preset['episode_probability']

    engine_mod.SUSTAINED_DEGRADATION_CYCLES = resolved_confirmation_cycles
    engine_mod.RETURN_CONFIRMATION_CYCLES = resolved_confirmation_cycles

    rng = np.random.default_rng(seed)
    random.seed(seed)

    scenario = ScenarioGenerator(
        episode_probability=resolved_probability,
        confirmation_cycles=resolved_confirmation_cycles,
        peak_hour_only=peak_hour_only,
        peak_hour_start=peak_hour_start,
        peak_hour_duration=peak_hour_duration,
        rng=rng,
    )
    engine = SyntheticBGPFailoverEngine(scenario)

    if not engine.ts_client:
        logger.error("❌ No se pudo conectar a TimescaleDB — abortando. "
                      "Verificar TIMESCALEDB_* en bgp_failover_engine_new.py")
        return

    mirror_rows = []
    if csv_mirror_path:
        original_insert = engine.ts_client.insert_bgp_metrics_new

        def _insert_and_mirror(metrics_dict):
            mirror_rows.append(dict(metrics_dict))
            return original_insert(metrics_dict)

        engine.ts_client.insert_bgp_metrics_new = _insert_and_mirror

    engine_mod.datetime = _FrozenDateTime

    logger.info("=" * 80)
    logger.info(f"🎲 Generando {cycles} ciclos sintéticos — escala: '{scale}'")
    logger.info("=" * 80)
    logger.info(f"   ⚠️ Datos ALEATORIOS para probar el pipeline — NO representativos de red real")
    logger.info(f"   Ventana de confirmación (SUSTAINED/RETURN_CONFIRMATION_CYCLES): {resolved_confirmation_cycles} ciclos "
                f"(~{resolved_confirmation_cycles * interval_seconds / 60:.1f} min a {interval_seconds}s/ciclo)")
    if peak_hour_only:
        logger.info(f"   Episodios restringidos a horario pico: {peak_hour_start:02d}:00-"
                     f"{(peak_hour_start + peak_hour_duration) % 24:02d}:00 "
                     f"({peak_hour_duration}h/día — supuesto, no confirmado por el ISP)")
        logger.info(f"   Tasa objetivo: {resolved_events_per_week} eventos/semana -> "
                     f"probabilidad calibrada por ciclo en pico: {resolved_probability:.6f} ({resolved_probability*100:.4f}%)")
    else:
        logger.info(f"   Probabilidad de episodio por ciclo (sin restricción horaria): {resolved_probability:.1%}")
    logger.info(f"   Intervalo entre ciclos: {interval_seconds}s | Inicio: {start_time.isoformat()}")
    logger.info(f"   Provider inicial (continúa desde BD): {engine.current_primary_provider}")
    logger.info(f"   Cycle_number inicial (continúa desde BD): {engine.cycle_count}")
    logger.info("=" * 80)

    current_time = start_time
    for i in range(cycles):
        _FrozenDateTime._fixed_now = current_time
        scenario.set_current_time(current_time)
        engine.run_cycle()
        current_time = current_time + timedelta(seconds=interval_seconds)

        if (i + 1) % max(1, cycles // 20) == 0:
            logger.info(f"   ✓ {i + 1}/{cycles} ciclos generados")

    logger.info("=" * 80)
    logger.info(f"✅ Generación completa: {cycles} ciclos insertados en bgp_metrics_new")
    logger.info(f"   Rango de tiempo simulado: {start_time.isoformat()} → {current_time.isoformat()}")
    span_weeks = (current_time - start_time).total_seconds() / (86400 * 7)
    logger.info(f"   Rango simulado: {span_weeks:.2f} semanas")
    logger.info("=" * 80)

    if csv_mirror_path and mirror_rows:
        import pandas as pd
        pd.DataFrame(mirror_rows).to_csv(csv_mirror_path, index=False)
        logger.info(f"📄 Espejo CSV guardado en: {csv_mirror_path}")

    logger.info("\n🎯 Próximos pasos:")
    logger.info("   1. Ejecutar feature_engine_incremental.py para derivar ml_features")
    logger.info("   2. Ejecutar train_logistic_regression.py para entrenar/comparar ambos modelos")
    if scale == 'realistic':
        logger.info("\n⚠️ Recordatorio para el informe: este dataset usa parámetros calibrados de")
        logger.info("   mejor esfuerzo (horario pico asumido en 4h/día, cadencia de medición del ISP")
        logger.info("   no confirmada) — declarar estos supuestos explícitamente si se citan los")
        logger.info("   resultados de esta corrida como evidencia.")


def main():
    parser = argparse.ArgumentParser(
        description='Genera datos sintéticos en bgp_metrics_new reutilizando el motor real de failover. '
                     'v2: soporta escala "lab" (rápida, para probar el pipeline) y "realistic" '
                     '(calibrada contra la operación real informada por el ISP).'
    )
    parser.add_argument('--cycles', type=int, default=300, help='Cantidad de ciclos a generar (default: 300)')
    parser.add_argument('--scale', choices=['lab', 'realistic'], default='lab',
                         help='Preset de calibración (default: lab).')
    parser.add_argument('--seed', type=int, default=42, help='Semilla de reproducibilidad')
    parser.add_argument('--interval-seconds', type=int, default=REAL_CYCLE_INTERVAL_SECONDS,
                         help=f'Segundos simulados entre ciclos (default: {REAL_CYCLE_INTERVAL_SECONDS})')
    parser.add_argument('--days-back', type=float, default=None,
                         help='Si se especifica, el primer ciclo empieza hace N días')
    parser.add_argument('--csv-mirror', type=str, default=None,
                         help='Ruta opcional para guardar además un CSV espejo de las filas insertadas')

    parser.add_argument('--events-per-week', type=float, default=None,
                         help='Tasa objetivo de eventos/semana (default del preset "realistic": 2.5)')
    parser.add_argument('--peak-hour-start', type=int, default=19,
                         help='Hora de inicio del horario pico, 0-23 (default: 19 — supuesto)')
    parser.add_argument('--peak-hour-duration', type=int, default=4,
                         help='Duración del horario pico en horas (default: 4 — supuesto)')
    parser.add_argument('--confirmation-cycles', type=int, default=None,
                         help='Ciclos de confirmación sostenida (override del preset)')
    parser.add_argument('--episode-probability', type=float, default=None,
                         help='Override manual directo de la probabilidad por ciclo')
    args = parser.parse_args()

    if args.days_back is not None:
        start_time = real_datetime_module.datetime.now(real_datetime_module.timezone.utc) - timedelta(days=args.days_back)
    else:
        total_span = timedelta(seconds=args.interval_seconds * args.cycles)
        start_time = real_datetime_module.datetime.now(real_datetime_module.timezone.utc) - total_span

    generate(
        cycles=args.cycles,
        scale=args.scale,
        seed=args.seed,
        start_time=start_time,
        interval_seconds=args.interval_seconds,
        csv_mirror_path=args.csv_mirror,
        events_per_week=args.events_per_week,
        peak_hour_start=args.peak_hour_start,
        peak_hour_duration=args.peak_hour_duration,
        confirmation_cycles=args.confirmation_cycles,
        episode_probability_override=args.episode_probability,
    )


if __name__ == '__main__':
    main()
