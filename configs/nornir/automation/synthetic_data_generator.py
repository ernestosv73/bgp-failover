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

✅ v4 — ruido de fondo con memoria temporal (Ornstein-Uhlenbeck). Antes cada
ciclo se muestreaba con np.random.normal() de forma INDEPENDIENTE — ruido
blanco, sin relación entre un ciclo y el siguiente. La latencia real no se
comporta así: si la red está algo congestionada ahora, es esperable que siga
algo congestionada 45s después, con una recuperación gradual, no saltos
i.i.d. Un Gaussian Process con kernel exponencial modela exactamente eso,
pero es computacionalmente inviable ciclo a ciclo a esta escala (matriz de
covarianza de tamaño N×N). El proceso de Ornstein-Uhlenbeck discretizado
(Euler-Maruyama) es la aproximación estándar: converge al mismo GP con
kernel exponencial, a costo O(1) por muestra — ver clase OUProcess.
Configurable vía --noise-model {iid,ou} (default: ou) y --ou-theta.

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

    # Volver al ruido i.i.d. anterior (para comparar / depurar):
    python3 synthetic_data_generator.py --scale lab --cycles 300 --noise-model iid
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import logging
import math
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

# ✅ v4 — velocidad de reversión a la media del proceso OU, en 1/ciclos.
# theta=1/3 implica una memoria característica de ~3 ciclos (~2.25 min a
# 45s/ciclo) — un valor de latencia "recuerda" haber estado alto/bajo
# durante unos pocos ciclos antes de volver a difuminarse hacia la media,
# consistente con congestión real que no aparece/desaparece instantáneamente
# pero tampoco persiste indefinidamente sin convertirse en un episodio.
DEFAULT_OU_THETA = 1.0 / 3.0

PEAK_SEVERITY_RANGE_DNS1 = (32, 60)
PEAK_SEVERITY_RANGE_DNS2 = (62, 90)
PEAK_JITTER_RANGE = (3, 12)

# ✅ Tipos de episodio considerados degradación "sostenida" real — los que
# nos interesan para el entrenamiento de detección de anomalías (a
# diferencia de spike/unstable/oscillating, que son transitorios/no
# necesariamente ameritan acción). Usado al poblar anomaly_ground_truth.
SUSTAINED_EPISODE_TYPES = {'step', 'slow_increase'}

GROUND_TRUTH_FLUSH_EVERY_CYCLES = 500  # cada cuántos ciclos se hace INSERT por lotes


def _insert_ground_truth_batch(conn, rows):
    """
    Inserta un lote de filas en anomaly_ground_truth. Devuelve la cantidad
    insertada, o None si falló (ej. la tabla no existe todavía porque no se
    corrió migration_ground_truth.sql) — en ese caso el llamador debe dejar
    de intentarlo para el resto de la corrida, no tiene sentido reintentar
    ciclo a ciclo si la tabla no existe.
    """
    if not rows:
        return 0
    try:
        from psycopg2.extras import execute_values
        cur = conn.cursor()
        execute_values(
            cur,
            "INSERT INTO anomaly_ground_truth "
            "(time, cycle_number, dns_target, episode_type, is_anomaly, is_sustained) VALUES %s",
            [(r['time'], r['cycle_number'], r['dns_target'], r['episode_type'],
              r['is_anomaly'], r['is_sustained']) for r in rows]
        )
        conn.commit()
        cur.close()
        return len(rows)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo insertar en anomaly_ground_truth: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None

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


class OUProcess:
    """
    Proceso de Ornstein-Uhlenbeck discretizado (Euler-Maruyama, dt=1 ciclo).

    Aproxima, a costo O(1) por muestra, un Gaussian Process con kernel
    exponencial — le da al ruido de fondo memoria/autocorrelación temporal
    en vez de ser blanco (i.i.d.). Converge a la MISMA distribución
    estacionaria (media, desvío) que el muestreo i.i.d. anterior, así que
    los rangos ya calibrados en BASELINE siguen siendo válidos sin
    recalibrar — lo único que cambia es la correlación entre ciclos
    consecutivos, no el rango de valores típicos.

    x_{t+1} = x_t + theta·(mu - x_t)·dt + sigma·√dt·ε_t,  ε_t ~ N(0,1)

    Con sigma = √(2·theta)·std_objetivo, la varianza estacionaria del
    proceso converge exactamente a std_objetivo² (ver derivación estándar
    de la varianza estacionaria de un proceso OU).
    """

    def __init__(self, mean, std, theta=DEFAULT_OU_THETA, dt=1.0, rng=None, floor=0.3):
        self.mean = mean
        self.theta = theta
        self.sigma = math.sqrt(2 * theta) * std
        self.dt = dt
        self.rng = rng or np.random.default_rng()
        self.floor = floor
        self.value = mean  # arranca en la media estacionaria

    def sample(self):
        noise = float(self.rng.normal(0, 1)) * self.sigma * math.sqrt(self.dt)
        self.value = self.value + self.theta * (self.mean - self.value) * self.dt + noise
        return max(self.floor, self.value)


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

    ✅ v3 — múltiples tipos de episodio (taxonomía de "Detecting Anomalies in
    Network Latency Time Series: From Statistical Filters to Machine
    Learning"). Antes solo existía un tipo ("step": rampa-meseta-rampa).
    Ahora cada episodio elige aleatoriamente uno de:
      - 'step'        (ya existía): degradación sostenida clásica — la que
                        debe confirmar failover tras N ciclos.
      - 'spike'        (nuevo): pico aislado de 1-2 ciclos, sin sostenerse.
                        Debe SER IGNORADO por el motor (prueba de que el
                        anti-flapping realmente filtra ruido transitorio).
      - 'unstable'     (nuevo): alta varianza SIN corrimiento de la media —
                        valida que cv_dns1/cv_dns2 capturen variabilidad
                        anómala que un umbral fijo sobre la media no vería.
      - 'oscillating'  (nuevo): alternancia periódica alto/bajo con período
                        MENOR a confirmation_cycles — caso adversarial para
                        el contador de degradación sostenida: cada ciclo
                        "bueno" intercalado debería resetearlo, así que
                        NUNCA debería disparar failover pese a cruzar el
                        umbral crítico repetidamente.
      - 'slow_increase' (nuevo): deriva GRADUAL hacia un valor alto a lo
                        largo de MUCHOS ciclos (rampa larga), a diferencia
                        de 'step' (rampa corta de 2-4 ciclos + meseta). El
                        paper la distingue explícitamente de 'step': *"latency
                        slowly drifts to a higher value over a long period of
                        time, rather than a sudden jump"*. Sirve para
                        validar si las features de tendencia/derivada del Z
                        detectan la degradación DURANTE la deriva, antes de
                        que el valor ya esté en su punto más alto.
    """

    EPISODE_TYPES = ('step', 'spike', 'unstable', 'oscillating', 'slow_increase')
    DEFAULT_EPISODE_TYPE_WEIGHTS = {
        'step': 0.45, 'spike': 0.15, 'unstable': 0.15,
        'oscillating': 0.10, 'slow_increase': 0.15,
    }

    def __init__(self, episode_probability, confirmation_cycles=3,
                 peak_hour_only=False, peak_hour_start=19, peak_hour_duration=4,
                 episode_type_weights=None, noise_model='ou', ou_theta=DEFAULT_OU_THETA,
                 seed=None, rng=None):
        self.episode_probability = episode_probability
        self.confirmation_cycles = confirmation_cycles
        self.peak_hour_only = peak_hour_only
        self.peak_hour_start = peak_hour_start
        self.peak_hour_duration = peak_hour_duration
        self.episode_type_weights = episode_type_weights or dict(self.DEFAULT_EPISODE_TYPE_WEIGHTS)

        # ✅ v4.1 fix — streams de RNG INDEPENDIENTES por proceso lógico.
        # ANTES: un único self.rng compartido entre los 6 procesos OU Y la
        # lógica de episodios (elegir target/tipo/severidad/duración). Como
        # el arranque de un episodio consume una cantidad VARIABLE de
        # números aleatorios, eso desfasaba la secuencia que después
        # alimentaba al proceso OU de 'peer' — y como OU tiene memoria (a
        # diferencia del ruido i.i.d. anterior), ese desfase NO se disolvía,
        # quedaba incorporado en la trayectoria de peer de ahí en más.
        # Resultado medido: 'peer' aparecía como la 2da feature más
        # importante (19%) en XGBoost pese a que NINGÚN evento de failover
        # real fue causado por peer (confirmado contra bgp_failover_events:
        # 100% de los change_reason mencionan solo dns1/dns2) — una
        # correlación espuria por compartir el generador aleatorio, no una
        # relación real entre las métricas.
        # Ahora cada uno de los 6 procesos de línea base (peer/dns1/dns2 ×
        # avg/stddev) tiene su PROPIO stream, más uno separado para la
        # lógica de episodios — derivados todos de la misma semilla maestra
        # (numpy.random.SeedSequence.spawn), así que la corrida sigue siendo
        # 100% reproducible con --seed, pero las series quedan
        # estadísticamente independientes entre sí de verdad.
        if seed is not None:
            seed_seq = np.random.SeedSequence(seed)
        elif rng is not None:
            seed_seq = np.random.SeedSequence(int(rng.integers(0, 2**31 - 1)))
        else:
            seed_seq = np.random.SeedSequence()

        baseline_keys = list(BASELINE.keys())
        child_seeds = seed_seq.spawn(len(baseline_keys) + 1)
        self.rng = np.random.default_rng(child_seeds[0])  # lógica de episodios
        self._baseline_rngs = {
            key: np.random.default_rng(child_seeds[i + 1]) for i, key in enumerate(baseline_keys)
        }

        # ✅ v4 — ruido de fondo con memoria (OU) vs. i.i.d. (comportamiento
        # anterior, disponible para comparar/depurar vía --noise-model iid)
        self.noise_model = noise_model
        if noise_model not in ('iid', 'ou'):
            raise ValueError(f"noise_model debe ser 'iid' u 'ou', recibido: {noise_model}")
        self.ou_theta = ou_theta
        self.ou_processes = {
            key: OUProcess(mean, std, theta=ou_theta, rng=self._baseline_rngs[key])
            for key, (mean, std) in BASELINE.items()
        } if noise_model == 'ou' else {}

        self.current_hour = None

        self.state = 'NORMAL'
        self.episode_type = None
        self.target = None
        self.state_cycles_left = 0
        self.peak_value = None
        self.ramp_start_value = None
        self.ramp_progress = 0
        self.ramp_total = 0

        # ✅ Metadata de verdad de terreno (ver next_metrics) — refleja qué
        # episodio estaba activo DURANTE el último next_metrics() devuelto.
        self.last_active_target = None
        self.last_active_episode_type = None

        # Estado específico de 'oscillating'
        self.osc_period = 1
        self.osc_counter = 0
        self.osc_phase_high = True

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
        if self.noise_model == 'ou':
            return self.ou_processes[key].sample()
        # Comportamiento i.i.d. anterior (v1-v3), disponible para comparar —
        # ahora también con stream propio por key (v4.1), no el compartido.
        mean, std = BASELINE[key]
        return max(0.3, float(self._baseline_rngs[key].normal(mean, std)))

    def _start_episode(self):
        self.target = self.rng.choice(['dns1', 'dns2'], p=[0.6, 0.4])
        if self.target == 'dns1':
            self.peak_value = float(self.rng.uniform(*PEAK_SEVERITY_RANGE_DNS1))
        else:
            self.peak_value = float(self.rng.uniform(*PEAK_SEVERITY_RANGE_DNS2))

        types = list(self.episode_type_weights.keys())
        probs = list(self.episode_type_weights.values())
        self.episode_type = self.rng.choice(types, p=probs)

        if self.episode_type == 'step':
            self.state = 'RAMP_UP'
            self.ramp_total = int(self.rng.integers(2, 4))
            self.ramp_progress = 0
            self.ramp_start_value = self._sample_baseline(f'{self.target}_avg')
            self.state_cycles_left = self.ramp_total

        elif self.episode_type == 'slow_increase':
            # Deriva gradual: rampa MUCHO más larga que 'step' (varias veces
            # confirmation_cycles), meseta corta al llegar arriba. Reutiliza
            # los mismos estados RAMP_UP/PLATEAU/RAMP_DOWN — la interpolación
            # es idéntica, solo cambia cuánto dura cada tramo (ver el branch
            # por episode_type al cerrar RAMP_UP más abajo).
            self.state = 'RAMP_UP'
            self.ramp_total = int(self.rng.integers(self.confirmation_cycles * 3, self.confirmation_cycles * 6 + 1))
            self.ramp_progress = 0
            self.ramp_start_value = self._sample_baseline(f'{self.target}_avg')
            self.state_cycles_left = self.ramp_total

        elif self.episode_type == 'spike':
            # Pico aislado: 1-2 ciclos, sin rampa — salto directo y vuelta directa.
            self.state = 'SPIKE_ACTIVE'
            self.state_cycles_left = int(self.rng.integers(1, 3))

        elif self.episode_type == 'unstable':
            # Alta varianza SOSTENIDA, sin mover la media — misma duración
            # que un plateau normal, para que sea comparable en el tiempo.
            self.state = 'UNSTABLE_ACTIVE'
            self.state_cycles_left = int(
                self.rng.integers(self.confirmation_cycles + 2, self.confirmation_cycles + 15)
            )

        elif self.episode_type == 'oscillating':
            # Alternancia alto/bajo con período MENOR a confirmation_cycles,
            # sostenida varios períodos completos — debería resetear el
            # contador de degradación en cada ciclo "bueno" intercalado.
            self.state = 'OSCILLATING_ACTIVE'
            self.osc_period = max(1, int(self.rng.integers(1, max(2, self.confirmation_cycles // 3))))
            self.state_cycles_left = self.osc_period * int(self.rng.integers(6, 12))
            self.osc_counter = 0
            self.osc_phase_high = True

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
                self.last_active_target = None
                self.last_active_episode_type = None
                return m

        # ✅ Captura de metadata para la tabla de verdad de terreno (ground
        # truth) — DEBE leerse acá, ANTES de que el manejo de estado de abajo
        # pueda resetear self.target/self.episode_type a None si este es el
        # último ciclo del episodio. Si se leyera después del if/elif,
        # la fila del ciclo que efectivamente cierra el episodio (ej. el
        # último ciclo de RAMP_DOWN, o el único ciclo de un spike de 1) se
        # grabaría incorrectamente como "normal" pese a que sus valores
        # numéricos SÍ reflejan la anomalía.
        self.last_active_target = self.target
        self.last_active_episode_type = self.episode_type

        if self.state == 'RAMP_UP':
            self.ramp_progress += 1
            frac = self.ramp_progress / max(1, self.ramp_total)
            interp = self.ramp_start_value + (self.peak_value - self.ramp_start_value) * frac
            m[f'{self.target}_avg'] = interp
            m[f'{self.target}_stddev'] = max(0.2, self._sample_baseline(f'{self.target}_stddev') * (1 + frac))
            self.state_cycles_left -= 1
            if self.state_cycles_left <= 0:
                self.state = 'PLATEAU'
                if self.episode_type == 'slow_increase':
                    # Lo sostenido acá es la RAMPA, no la meseta — meseta corta.
                    self.state_cycles_left = int(self.rng.integers(2, max(3, self.confirmation_cycles)))
                else:
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
                self.episode_type = None

        elif self.state == 'SPIKE_ACTIVE':
            # Salto directo al pico, sin rampa — y vuelta directa al terminar.
            m[f'{self.target}_avg'] = self.peak_value
            m[f'{self.target}_stddev'] = float(self.rng.uniform(*PEAK_JITTER_RANGE))
            self.state_cycles_left -= 1
            if self.state_cycles_left <= 0:
                self.state = 'NORMAL'
                self.target = None
                self.episode_type = None

        elif self.state == 'UNSTABLE_ACTIVE':
            # La MEDIA se mantiene cerca del baseline — lo que sube es la
            # dispersión ciclo a ciclo (tanto del valor como del jitter).
            baseline_mean, _ = BASELINE[f'{self.target}_avg']
            noisy_val = baseline_mean + float(self.rng.normal(0, baseline_mean * 0.9))
            m[f'{self.target}_avg'] = max(0.3, noisy_val)
            m[f'{self.target}_stddev'] = float(self.rng.uniform(*PEAK_JITTER_RANGE))
            self.state_cycles_left -= 1
            if self.state_cycles_left <= 0:
                self.state = 'NORMAL'
                self.target = None
                self.episode_type = None

        elif self.state == 'OSCILLATING_ACTIVE':
            self.osc_counter += 1
            if self.osc_counter >= self.osc_period:
                self.osc_counter = 0
                self.osc_phase_high = not self.osc_phase_high

            if self.osc_phase_high:
                m[f'{self.target}_avg'] = self.peak_value
                m[f'{self.target}_stddev'] = float(self.rng.uniform(*PEAK_JITTER_RANGE))
            else:
                m[f'{self.target}_avg'] = self._sample_baseline(f'{self.target}_avg')

            self.state_cycles_left -= 1
            if self.state_cycles_left <= 0:
                self.state = 'NORMAL'
                self.target = None
                self.episode_type = None

        return m


class SyntheticBGPFailoverEngine(BGPFailoverEngine):
    def __init__(self, scenario_generator: ScenarioGenerator):
        self.scenario_generator = scenario_generator
        self.last_ground_truth = None  # (target_dns o None, episode_type o None) del último ciclo medido
        super().__init__()

    def measure_provider_latency(self):
        raw = self.scenario_generator.next_metrics()
        metrics = LatencyMetrics(
            peer_avg=raw['peer_avg'], peer_loss=0.0, peer_stddev=raw['peer_stddev'],
            dns1_avg=raw['dns1_avg'], dns1_loss=0.0, dns1_stddev=raw['dns1_stddev'],
            dns2_avg=raw['dns2_avg'], dns2_loss=0.0, dns2_stddev=raw['dns2_stddev'],
        )
        # ✅ Capturar la metadata de verdad de terreno de ESTE ciclo — ver
        # comentario en ScenarioGenerator.next_metrics() sobre por qué se lee
        # de last_active_target/last_active_episode_type y no de
        # self.scenario_generator.target/episode_type directamente (esos
        # pueden haberse reseteado ya si este era el último ciclo del episodio).
        self.last_ground_truth = (
            self.scenario_generator.last_active_target,
            self.scenario_generator.last_active_episode_type,
        )
        logging.info(
            f"🎲 Sintético — Peer: {metrics.peer_avg:.2f}ms | "
            f"DNS1: {metrics.dns1_avg:.2f}ms (jit {metrics.dns1_stddev:.2f}) | "
            f"DNS2: {metrics.dns2_avg:.2f}ms (jit {metrics.dns2_stddev:.2f}) "
            f"[escenario: {self.scenario_generator.state} | tipo: {self.scenario_generator.episode_type}]"
        )
        return metrics


def compute_calibrated_probability(events_per_week, interval_seconds, peak_hour_duration):
    cycles_per_day = 86400 / interval_seconds
    peak_cycles_per_day = cycles_per_day * (peak_hour_duration / 24)
    peak_cycles_per_week = peak_cycles_per_day * 7
    return events_per_week / peak_cycles_per_week


def generate(cycles, scale, seed, start_time, interval_seconds, csv_mirror_path,
             events_per_week=None, peak_hour_start=19, peak_hour_duration=4,
             confirmation_cycles=None, episode_probability_override=None,
             episode_type_weights=None, noise_model='ou', ou_theta=DEFAULT_OU_THETA):
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

    random.seed(seed)

    scenario = ScenarioGenerator(
        episode_probability=resolved_probability,
        confirmation_cycles=resolved_confirmation_cycles,
        peak_hour_only=peak_hour_only,
        peak_hour_start=peak_hour_start,
        peak_hour_duration=peak_hour_duration,
        episode_type_weights=episode_type_weights,
        noise_model=noise_model,
        ou_theta=ou_theta,
        seed=seed,
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
    logger.info(f"   Mezcla de tipos de episodio: {scenario.episode_type_weights}")
    logger.info(f"   Modelo de ruido de fondo: {noise_model}"
                + (f" (theta={ou_theta:.3f}, memoria≈{1/ou_theta:.1f} ciclos)" if noise_model == 'ou' else ""))
    logger.info(f"   Provider inicial (continúa desde BD): {engine.current_primary_provider}")
    logger.info(f"   Cycle_number inicial (continúa desde BD): {engine.cycle_count}")
    logger.info("=" * 80)

    ground_truth_buffer = []
    ground_truth_inserted = 0
    ground_truth_failed = False

    current_time = start_time
    for i in range(cycles):
        _FrozenDateTime._fixed_now = current_time
        scenario.set_current_time(current_time)
        engine.run_cycle()

        # ✅ Verdad de terreno: 2 filas por ciclo (dns1, dns2) — is_anomaly=True
        # solo en la fila cuyo dns_target coincide con el episodio activo ESE
        # ciclo (ver last_ground_truth, capturado antes de cualquier reset de
        # fin-de-episodio dentro de ScenarioGenerator.next_metrics()).
        if not ground_truth_failed:
            cycle_number_just_processed = engine.cycle_count - 1
            active_target, active_type = engine.last_ground_truth or (None, None)
            for dns_name in ('dns1', 'dns2'):
                is_anomaly = (active_target == dns_name)
                ep_type = active_type if is_anomaly else None
                ground_truth_buffer.append({
                    'time': current_time,
                    'cycle_number': cycle_number_just_processed,
                    'dns_target': dns_name,
                    'episode_type': ep_type,
                    'is_anomaly': is_anomaly,
                    'is_sustained': bool(is_anomaly and ep_type in SUSTAINED_EPISODE_TYPES),
                })

        current_time = current_time + timedelta(seconds=interval_seconds)

        if len(ground_truth_buffer) >= GROUND_TRUTH_FLUSH_EVERY_CYCLES * 2:
            inserted = _insert_ground_truth_batch(engine.ts_client.conn, ground_truth_buffer)
            if inserted is None:
                ground_truth_failed = True
                logger.warning("⚠️ Se desactivó el registro de verdad de terreno para el resto de esta corrida "
                                "(¿corriste migration_ground_truth.sql?)")
            else:
                ground_truth_inserted += inserted
            ground_truth_buffer = []

        if (i + 1) % max(1, cycles // 20) == 0:
            logger.info(f"   ✓ {i + 1}/{cycles} ciclos generados")

    if not ground_truth_failed and ground_truth_buffer:
        inserted = _insert_ground_truth_batch(engine.ts_client.conn, ground_truth_buffer)
        if inserted:
            ground_truth_inserted += inserted

    logger.info("=" * 80)
    logger.info(f"✅ Generación completa: {cycles} ciclos insertados en bgp_metrics_new")
    if not ground_truth_failed:
        logger.info(f"✅ Verdad de terreno: {ground_truth_inserted} filas insertadas en anomaly_ground_truth "
                     f"(2 por ciclo — una por cada DNS)")
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
    parser.add_argument('--episode-type-weights', type=str, default=None,
                         help='JSON con la mezcla de tipos de episodio, ej. '
                              '\'{"step":0.55,"spike":0.15,"unstable":0.15,"oscillating":0.15}\'. '
                              'Default: esa misma mezcla (ver ScenarioGenerator.DEFAULT_EPISODE_TYPE_WEIGHTS).')
    parser.add_argument('--noise-model', choices=['iid', 'ou'], default='ou',
                         help='Modelo del ruido de fondo (default: ou). '
                              '"ou" = Ornstein-Uhlenbeck (con memoria/autocorrelación temporal, '
                              'aproxima un Gaussian Process). "iid" = muestreo independiente '
                              'ciclo a ciclo (comportamiento v1-v3, sin memoria — para comparar).')
    parser.add_argument('--ou-theta', type=float, default=DEFAULT_OU_THETA,
                         help=f'Velocidad de reversión a la media del proceso OU, en 1/ciclos '
                              f'(default: {DEFAULT_OU_THETA:.3f} ≈ memoria de 3 ciclos). '
                              f'Valores más chicos = memoria más larga.')
    args = parser.parse_args()

    if args.days_back is not None:
        start_time = real_datetime_module.datetime.now(real_datetime_module.timezone.utc) - timedelta(days=args.days_back)
    else:
        total_span = timedelta(seconds=args.interval_seconds * args.cycles)
        start_time = real_datetime_module.datetime.now(real_datetime_module.timezone.utc) - total_span

    episode_type_weights = None
    if args.episode_type_weights:
        import json
        episode_type_weights = json.loads(args.episode_type_weights)

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
        episode_type_weights=episode_type_weights,
        noise_model=args.noise_model,
        ou_theta=args.ou_theta,
    )


if __name__ == '__main__':
    main()
