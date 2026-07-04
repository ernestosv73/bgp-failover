#!/usr/bin/env python3
"""
data_generator.py - Generador de Datos Sintéticos para BGP Failover ML
Genera datos estadísticamente coherentes con los patrones reales observados
en bgp_metrics.csv y ml_features.csv

✅ ESCENARIOS SIMULADOS:
├─ Normal (95%): Ambos providers estables
├─ Degradación súbita (2%): Spike de latencia
├─ Degradación gradual (1%): Aumento progresivo
├─ Spike de pérdida (1%): Pérdida momentánea
├─ Degradación DNS (0.5%): Latencia alta en DNS
└─ Degradación ambos (0.5%): Ambos providers degradados

✅ LÓGICA DE FAILOVER AUTOMÁTICO:
├─ Detección combinada: Z-score + Absolute + Relative
├─ Failover por diferencia de score > 5 puntos durante 3 ciclos
├─ Failover por anomalía combinada (degraded/critical) durante 3 ciclos
└─ Frecuencia esperada: 2-3 failovers por semana

✅ COHERENCIA CON LA SOLUCIÓN:
├─ Fórmula de scoring: score = (peer×0.4) + (dns×0.6) + (loss×0.5) + (jitter×0.5)
├─ Umbrales de detección combinada
├─ Lógica de failover (3 ciclos sostenidos)
├─ Estructura de ml_features (41 columnas)
└─ Target failover_event (1 registro por failover)

USO:
    python3 data_generator.py --cycles 10000 --output /tmp/synthetic_data.csv
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURACIÓN ===
SCORING_WEIGHTS = {
    'peer_latency': 0.4,
    'dns_latency': 0.6,
    'loss': 0.5,
    'jitter': 0.5
}

SUSTAINED_DEGRADATION_CYCLES = 3
SWITCH_MARGIN = 5
MAX_LATENCY = 50.0

# Umbrales de detección combinada
Z_SCORE_THRESHOLDS = {'normal': 2.0, 'warning': 2.5, 'degraded': 3.0, 'critical': 3.5}
ABSOLUTE_LATENCY_THRESHOLDS = {
    'peer_warning': 12.0, 'peer_degraded': 15.0, 'peer_critical': 25.0,
    'dns_warning': 15.0, 'dns_degraded': 20.0, 'dns_critical': 30.0
}
RELATIVE_DIFF_THRESHOLDS = {'warning': 5.0, 'degraded': 10.0, 'critical': 15.0}

# ✅ DISTRIBUCIÓN DE ESCENARIOS (ajustada para 2-3 failovers/semana)
SCENARIO_PROBABILITIES = {
    'normal': 0.950,              # 95% normal
    'degradation_sudden': 0.020,  # 2% degradación súbita
    'degradation_gradual': 0.010, # 1% degradación gradual
    'loss_spike': 0.010,          # 1% spike de pérdida
    'dns_degradation': 0.005,     # 0.5% degradación DNS
    'degradation_both': 0.005     # 0.5% degradación ambos
}


class BGPDataGenerator:
    """Generador de datos sintéticos para BGP Failover"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.history = {'PROVIDER1': [], 'PROVIDER2': []}
        self.rolling_window = 10
        self.degradation_counter = 0
        self.current_provider = 'PROVIDER1'
        self.cycle_count = 0
        
        # ✅ Estadísticas de generación
        self.stats = {
            'total_cycles': 0,
            'automatic_failovers': 0,
            'degradations_detected': 0,
            'degradations_recovered': 0
        }
    
    def _generate_normal_metrics(self, provider):
        """Genera métricas normales (latencia baja, sin pérdida)"""
        peer_latency = np.random.normal(5.0, 1.5)
        dns_latency = np.random.normal(5.0, 1.5)
        peer_jitter = np.random.normal(2.0, 0.5)
        dns_jitter = np.random.normal(2.0, 0.5)
        
        # Asegurar valores positivos
        peer_latency = max(1.0, peer_latency)
        dns_latency = max(1.0, dns_latency)
        peer_jitter = max(0.1, peer_jitter)
        dns_jitter = max(0.1, dns_jitter)
        
        return {
            'peer_latency_ms': peer_latency,
            'dns_latency_ms': dns_latency,
            'peer_loss_pct': 0.0,
            'dns_loss_pct': 0.0,
            'peer_jitter_ms': peer_jitter,
            'dns_jitter_ms': dns_jitter
        }
    
    def _generate_degraded_metrics(self, provider, severity='warning'):
        """Genera métricas degradadas según severidad"""
        # Degradación en DNS (simulación correcta, no en peer)
        if severity == 'warning':
            dns_latency = np.random.normal(18.0, 2.0)
            peer_latency = np.random.normal(5.0, 1.5)
        elif severity == 'degraded':
            dns_latency = np.random.normal(25.0, 3.0)
            peer_latency = np.random.normal(6.0, 1.5)
        elif severity == 'critical':
            dns_latency = np.random.normal(35.0, 4.0)
            peer_latency = np.random.normal(7.0, 1.5)
        else:
            dns_latency = np.random.normal(15.0, 2.0)
            peer_latency = np.random.normal(5.0, 1.5)
        
        peer_jitter = np.random.normal(3.0, 1.0)
        dns_jitter = np.random.normal(4.0, 1.5)
        
        return {
            'peer_latency_ms': max(1.0, peer_latency),
            'dns_latency_ms': max(1.0, dns_latency),
            'peer_loss_pct': 0.0,
            'dns_loss_pct': 0.0,
            'peer_jitter_ms': max(0.1, peer_jitter),
            'dns_jitter_ms': max(0.1, dns_jitter)
        }
    
    def _generate_loss_spike(self, provider):
        """Genera spike de pérdida de paquetes"""
        metrics = self._generate_normal_metrics(provider)
        metrics['peer_loss_pct'] = np.random.uniform(5.0, 15.0)
        metrics['dns_loss_pct'] = np.random.uniform(5.0, 15.0)
        return metrics
    
    def _calculate_score(self, metrics):
        """Calcula score usando la fórmula actual"""
        weighted_latency = (
            metrics['peer_latency_ms'] * SCORING_WEIGHTS['peer_latency'] +
            metrics['dns_latency_ms'] * SCORING_WEIGHTS['dns_latency']
        )
        loss_penalty = (
            (metrics['peer_loss_pct'] + metrics['dns_loss_pct']) / 2 * SCORING_WEIGHTS['loss']
        )
        jitter_penalty = (
            (metrics['peer_jitter_ms'] + metrics['dns_jitter_ms']) / 2 * SCORING_WEIGHTS['jitter']
        )
        return weighted_latency + loss_penalty + jitter_penalty
    
    def _calculate_rolling_stats(self, provider):
        """Calcula rolling statistics (mean, std, p95)"""
        history = self.history[provider][-self.rolling_window:]
        
        if len(history) < 3:
            return {'mean': 0.0, 'std': 0.0, 'p95': 0.0}
        
        latencies = [m['peer_latency_ms'] for m in history]
        return {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'p95': float(np.percentile(latencies, 95))
        }
    
    def _calculate_z_score(self, current_value, rolling_stats):
        """Calcula Z-score"""
        if rolling_stats['std'] < 0.001:
            return 0.0
        return (current_value - rolling_stats['mean']) / rolling_stats['std']
    
    def _classify_z_score_severity(self, z_score):
        """Clasifica severidad del Z-score"""
        if z_score >= Z_SCORE_THRESHOLDS['critical']:
            return 'critical'
        elif z_score >= Z_SCORE_THRESHOLDS['degraded']:
            return 'degraded'
        elif z_score >= Z_SCORE_THRESHOLDS['warning']:
            return 'warning'
        return 'normal'
    
    def _classify_absolute_severity(self, metrics):
        """Clasifica severidad absoluta"""
        peer = metrics['peer_latency_ms']
        dns = metrics['dns_latency_ms']
        
        if peer >= ABSOLUTE_LATENCY_THRESHOLDS['peer_critical'] or \
           dns >= ABSOLUTE_LATENCY_THRESHOLDS['dns_critical']:
            return 'critical'
        elif peer >= ABSOLUTE_LATENCY_THRESHOLDS['peer_degraded'] or \
             dns >= ABSOLUTE_LATENCY_THRESHOLDS['dns_degraded']:
            return 'degraded'
        elif peer >= ABSOLUTE_LATENCY_THRESHOLDS['peer_warning'] or \
             dns >= ABSOLUTE_LATENCY_THRESHOLDS['dns_warning']:
            return 'warning'
        return 'normal'
    
    def _classify_relative_severity(self, relative_diff):
        """Clasifica severidad relativa"""
        if relative_diff >= RELATIVE_DIFF_THRESHOLDS['critical']:
            return 'critical'
        elif relative_diff >= RELATIVE_DIFF_THRESHOLDS['degraded']:
            return 'degraded'
        elif relative_diff >= RELATIVE_DIFF_THRESHOLDS['warning']:
            return 'warning'
        return 'normal'
    
    def _classify_combined_severity(self, z_sev, abs_sev, rel_sev):
        """Clasifica severidad combinada (máximo de las 3)"""
        severity_levels = {'normal': 0, 'warning': 1, 'degraded': 2, 'critical': 3}
        max_level = max(
            severity_levels[z_sev],
            severity_levels[abs_sev],
            severity_levels[rel_sev]
        )
        return [k for k, v in severity_levels.items() if v == max_level][0]
    
    def _generate_scenario(self, cycle_num, total_cycles):
        """Genera un escenario para un ciclo específico"""
        # Distribución de escenarios
        rand = random.random()
        cumulative = 0.0
        
        for scenario, probability in SCENARIO_PROBABILITIES.items():
            cumulative += probability
            if rand < cumulative:
                if scenario == 'normal':
                    return {
                        'scenario': 'normal',
                        'provider1_metrics': self._generate_normal_metrics('PROVIDER1'),
                        'provider2_metrics': self._generate_normal_metrics('PROVIDER2')
                    }
                elif scenario == 'degradation_sudden':
                    severity = random.choice(['warning', 'degraded', 'critical'])
                    return {
                        'scenario': 'degradation_sudden',
                        'provider1_metrics': self._generate_degraded_metrics('PROVIDER1', severity),
                        'provider2_metrics': self._generate_normal_metrics('PROVIDER2')
                    }
                elif scenario == 'degradation_gradual':
                    # Degradación progresiva a lo largo de varios ciclos
                    progress = (cycle_num % 5) / 5.0  # 0.0 a 0.8
                    dns_latency = 10.0 + (progress * 25.0)  # 10ms a 35ms
                    
                    metrics_p1 = self._generate_normal_metrics('PROVIDER1')
                    metrics_p1['dns_latency_ms'] = dns_latency
                    
                    return {
                        'scenario': 'degradation_gradual',
                        'provider1_metrics': metrics_p1,
                        'provider2_metrics': self._generate_normal_metrics('PROVIDER2')
                    }
                elif scenario == 'loss_spike':
                    return {
                        'scenario': 'loss_spike',
                        'provider1_metrics': self._generate_loss_spike('PROVIDER1'),
                        'provider2_metrics': self._generate_normal_metrics('PROVIDER2')
                    }
                elif scenario == 'dns_degradation':
                    metrics_p1 = self._generate_normal_metrics('PROVIDER1')
                    metrics_p1['dns_latency_ms'] = np.random.normal(25.0, 3.0)
                    return {
                        'scenario': 'dns_degradation',
                        'provider1_metrics': metrics_p1,
                        'provider2_metrics': self._generate_normal_metrics('PROVIDER2')
                    }
                elif scenario == 'degradation_both':
                    return {
                        'scenario': 'degradation_both',
                        'provider1_metrics': self._generate_degraded_metrics('PROVIDER1', 'degraded'),
                        'provider2_metrics': self._generate_degraded_metrics('PROVIDER2', 'warning')
                    }
        
        # Fallback a normal
        return {
            'scenario': 'normal',
            'provider1_metrics': self._generate_normal_metrics('PROVIDER1'),
            'provider2_metrics': self._generate_normal_metrics('PROVIDER2')
        }
    
    def _simulate_failover_logic(self, score_p1, score_p2, combined_sev_p1):
        """
        ✅ Simula la lógica de failover AUTOMÁTICO del motor BGP
        Basado en detección combinada y diferencia de score
        """
        score_diff = score_p1 - score_p2
        has_severe_anomaly = combined_sev_p1 in ['degraded', 'critical']
        
        # ✅ Lógica de failover automático
        if score_diff > SWITCH_MARGIN or has_severe_anomaly:
            self.degradation_counter += 1
            self.stats['degradations_detected'] += 1
            
            if self.degradation_counter >= SUSTAINED_DEGRADATION_CYCLES:
                # ✅ Failover ejecutado automáticamente
                self.stats['automatic_failovers'] += 1
                self.degradation_counter = 0
                self.current_provider = 'PROVIDER2' if self.current_provider == 'PROVIDER1' else 'PROVIDER1'
                return True, 0
            
            return False, self.degradation_counter
        else:
            # ✅ Degradación terminó sin failover
            if self.degradation_counter > 0:
                self.stats['degradations_recovered'] += 1
            self.degradation_counter = 0
            return False, 0
    
    def generate_cycle(self, cycle_num, total_cycles, base_time):
        """Genera un ciclo completo (2 registros: uno por provider)"""
        scenario = self._generate_scenario(cycle_num, total_cycles)
        timestamp = base_time + timedelta(seconds=cycle_num * 30)
        
        # Calcular scores
        score_p1 = self._calculate_score(scenario['provider1_metrics'])
        score_p2 = self._calculate_score(scenario['provider2_metrics'])
        
        # Calcular rolling stats
        rolling_p1 = self._calculate_rolling_stats('PROVIDER1')
        rolling_p2 = self._calculate_rolling_stats('PROVIDER2')
        
        # Calcular Z-scores
        z_score_p1 = self._calculate_z_score(
            scenario['provider1_metrics']['peer_latency_ms'], rolling_p1
        )
        z_score_p2 = self._calculate_z_score(
            scenario['provider2_metrics']['peer_latency_ms'], rolling_p2
        )
        
        # Clasificar severidades
        z_sev_p1 = self._classify_z_score_severity(z_score_p1)
        z_sev_p2 = self._classify_z_score_severity(z_score_p2)
        
        abs_sev_p1 = self._classify_absolute_severity(scenario['provider1_metrics'])
        abs_sev_p2 = self._classify_absolute_severity(scenario['provider2_metrics'])
        
        # Diferencia relativa (peer actual vs promedio del otro)
        other_avg_p1 = rolling_p2['mean'] if rolling_p2['mean'] > 0 else scenario['provider2_metrics']['peer_latency_ms']
        other_avg_p2 = rolling_p1['mean'] if rolling_p1['mean'] > 0 else scenario['provider1_metrics']['peer_latency_ms']
        
        relative_diff_p1 = scenario['provider1_metrics']['peer_latency_ms'] - other_avg_p1
        relative_diff_p2 = scenario['provider2_metrics']['peer_latency_ms'] - other_avg_p2
        
        rel_sev_p1 = self._classify_relative_severity(relative_diff_p1)
        rel_sev_p2 = self._classify_relative_severity(relative_diff_p2)
        
        # Severidad combinada
        combined_sev_p1 = self._classify_combined_severity(z_sev_p1, abs_sev_p1, rel_sev_p1)
        combined_sev_p2 = self._classify_combined_severity(z_sev_p2, abs_sev_p2, rel_sev_p2)
        
        # ✅ Simular failover automático
        provider_changed, degradation_cycle = self._simulate_failover_logic(
            score_p1, score_p2, combined_sev_p1
        )
        
        # Score difference
        score_diff_p1 = score_p1 - score_p2
        score_diff_p2 = score_p2 - score_p1
        
        # Generar registros
        records = []
        for provider, metrics, score, rolling, z_score, z_sev, abs_sev, rel_diff, rel_sev, combined_sev, score_diff_for_provider, alternative_score in [
            ('PROVIDER1', scenario['provider1_metrics'], score_p1, rolling_p1, z_score_p1, z_sev_p1, abs_sev_p1, relative_diff_p1, rel_sev_p1, combined_sev_p1, score_diff_p1, score_p2),
            ('PROVIDER2', scenario['provider2_metrics'], score_p2, rolling_p2, z_score_p2, z_sev_p2, abs_sev_p2, relative_diff_p2, rel_sev_p2, combined_sev_p2, score_diff_p2, score_p1)
        ]:
            # Quality index
            weighted_lat = (
                metrics['peer_latency_ms'] * SCORING_WEIGHTS['peer_latency'] +
                metrics['dns_latency_ms'] * SCORING_WEIGHTS['dns_latency']
            )
            total_loss = (metrics['peer_loss_pct'] + metrics['dns_loss_pct']) / 2
            quality_index = max(0, min(100, 100 - (
                (weighted_lat / MAX_LATENCY * 40) +
                (total_loss * 50) +
                (metrics['peer_jitter_ms'] / 10 * 10)
            )))
            
            # Temporal features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_business_hours = 9 <= hour < 17 and day_of_week < 5
            is_peak_traffic = (10 <= hour < 14) or (15 <= hour < 18)
            is_weekend = day_of_week >= 5
            
            # failover_event: solo 1 registro por failover (el provider que perdió)
            failover_event = 1 if (provider_changed and score_diff_for_provider > 0) else 0
            
            record = {
                'time': timestamp,
                'provider': provider,
                'peer_latency_ms': round(metrics['peer_latency_ms'], 2),
                'dns_latency_ms': round(metrics['dns_latency_ms'], 2),
                'peer_loss_pct': round(metrics['peer_loss_pct'], 2),
                'dns_loss_pct': round(metrics['dns_loss_pct'], 2),
                'peer_jitter_ms': round(metrics['peer_jitter_ms'], 2),
                'dns_jitter_ms': round(metrics['dns_jitter_ms'], 2),
                'score': round(score, 2),
                'latency_ratio': round(metrics['peer_latency_ms'] / (metrics['dns_latency_ms'] + 0.001), 4),
                'total_loss_pct': round(total_loss, 2),
                'quality_index': round(quality_index, 2),
                'latency_trend_5min': round(np.random.normal(0, 1), 2),
                'latency_trend_15min': round(np.random.normal(0, 0.5), 2),
                'latency_velocity': round(np.random.normal(0, 2), 2),
                'latency_acceleration': round(np.random.normal(0, 1), 2),
                'loss_spike_detected': metrics['peer_loss_pct'] > 5.0,
                'hour_of_day': hour,
                'day_of_week': day_of_week,
                'is_business_hours': is_business_hours,
                'is_peak_traffic': is_peak_traffic,
                'is_weekend': is_weekend,
                'provider_changes_last_hour': 1 if provider_changed else 0,
                'time_since_last_change_min': 0.0 if provider_changed else np.random.uniform(1, 60),
                'current_provider_score': round(score, 2),
                'alternative_provider_score': round(alternative_score, 2),
                'score_difference': round(score_diff_for_provider, 2),
                'margin_exceeds_threshold': abs(score_diff_for_provider) > SWITCH_MARGIN,
                'should_failover': 1 if provider_changed else 0,
                'degradation_cycle': degradation_cycle,
                'provider_changed': provider_changed,
                'z_score_peer': round(z_score, 2),
                'z_score_severity': z_sev,
                'rolling_mean': round(rolling['mean'], 2),
                'rolling_std': round(rolling['std'], 2),
                'rolling_p95': round(rolling['p95'], 2),
                'absolute_severity': abs_sev,
                'relative_diff_ms': round(rel_diff, 2),
                'relative_severity': rel_sev,
                'combined_severity': combined_sev,
                'is_combined_anomaly': combined_sev in ['degraded', 'critical'],
                'failover_event': failover_event
            }
            records.append(record)
        
        # Actualizar historial
        self.history['PROVIDER1'].append(scenario['provider1_metrics'])
        self.history['PROVIDER2'].append(scenario['provider2_metrics'])
        
        # Limitar historial
        if len(self.history['PROVIDER1']) > self.rolling_window:
            self.history['PROVIDER1'].pop(0)
            self.history['PROVIDER2'].pop(0)
        
        self.stats['total_cycles'] += 1
        
        return records
    
    def generate_dataset(self, num_cycles=10000, output_file='/tmp/synthetic_ml_features.csv'):
        """Genera dataset completo"""
        logger.info(f"🚀 Generando {num_cycles} ciclos sintéticos...")
        logger.info(f"📊 Distribución de escenarios:")
        for scenario, prob in SCENARIO_PROBABILITIES.items():
            logger.info(f"   {scenario}: {prob*100:.1f}%")
        
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        all_records = []
        
        for cycle in range(num_cycles):
            records = self.generate_cycle(cycle, num_cycles, base_time)
            all_records.extend(records)
            
            if (cycle + 1) % 2000 == 0:
                logger.info(f"   ✓ Ciclo {cycle + 1}/{num_cycles} generado")
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_records)
        
        # Estadísticas
        total_records = len(df)
        total_failovers = df['failover_event'].sum()
        unique_failovers = df[df['failover_event'] == 1]['time'].nunique()
        
        logger.info(f"\n📊 Dataset generado:")
        logger.info(f"   Total registros: {total_records}")
        logger.info(f"   Total failovers (should_failover=1): {df['should_failover'].sum()}")
        logger.info(f"   Failovers únicos (failover_event=1): {total_failovers}")
        logger.info(f"   Ciclos únicos con failover: {unique_failovers}")
        logger.info(f"   🤖 Failovers automáticos: {self.stats['automatic_failovers']}")
        logger.info(f"   Período: {df['time'].min()} a {df['time'].max()}")
        
        # ✅ Calcular frecuencia de failovers
        days_span = (df['time'].max() - df['time'].min()).days
        failovers_per_week = (total_failovers / days_span) * 7 if days_span > 0 else 0
        logger.info(f"\n📈 Frecuencia de failovers:")
        logger.info(f"   Días cubiertos: {days_span}")
        logger.info(f"   Failovers por semana: {failovers_per_week:.2f}")
        logger.info(f"   Referencia ISP real: 2-3 failovers/semana")
        
        # Guardar
        df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Dataset guardado en: {output_file}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Generador de datos sintéticos para BGP Failover ML')
    parser.add_argument('--cycles', type=int, default=10000, help='Número de ciclos a generar')
    parser.add_argument('--output', type=str, default='/tmp/synthetic_ml_features.csv', help='Archivo de salida')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reproducibilidad')
    args = parser.parse_args()
    
    generator = BGPDataGenerator(seed=args.seed)
    df = generator.generate_dataset(num_cycles=args.cycles, output_file=args.output)
    
    logger.info(f"\n🎯 Próximos pasos:")
    logger.info(f"   1. Cargar datos en TimescaleDB:")
    logger.info(f"      psql -h timescaledb -U bgp_app -d bgp_failover_db -c \"\\copy ml_features FROM '{args.output}' CSV HEADER\"")
    logger.info(f"   2. Re-entrenar XGBoost:")
    logger.info(f"      python3 train_from_ml_features.py")
    logger.info(f"   3. Analizar feature importance y pesos optimizados")


if __name__ == '__main__':
    main()
