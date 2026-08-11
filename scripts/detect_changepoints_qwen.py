import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURACIÓN DE CONEXIÓN A BASE DE DATOS
# ==========================================
DB_CONFIG = {
    'user': 'bgp_app',
    'password': 'bgp_app_password',
    'host': 'timescaledb',
    'port': '5432',
    'database': 'bgp_failover_db'
}

DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

# ==========================================
# 2. CONFIGURACIÓN DE UMBRALES CRÍTICOS
# ==========================================
THRESHOLDS = {
    'dns1': {'warning': 15.0, 'critical': 30.0},
    'dns2': {'warning': 30.0, 'critical': 60.0}
}

# ==========================================
# 3. CARGA DE DATOS DESDE TIMESCALEDB
# ==========================================
def load_data_from_db(query=None):
    if query is None:
        query = """
            SELECT time, cycle_number,
                dns1_latency_ms, dns1_jitter_ms, dns1_loss_pct,
                dns2_latency_ms, dns2_jitter_ms, dns2_loss_pct
            FROM link_health_metrics ORDER BY time ASC;
        """
    print(f"Conectando a {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    try:
        df = pd.read_sql(query, con=engine)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        metric_columns = ['dns1_latency_ms', 'dns1_jitter_ms', 'dns1_loss_pct', 
                          'dns2_latency_ms', 'dns2_jitter_ms', 'dns2_loss_pct']
        df[metric_columns] = df[metric_columns].apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='linear', limit_direction='forward')
        
        print(f"✅ Datos cargados exitosamente. Total de registros: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Error al conectar o consultar la base de datos: {e}")
        return None

# ==========================================
# 4. MOTOR DE DETECCIÓN DE CHANGEPOINTS (PELT)
# ==========================================
def detect_changepoints_pelt(signal, penalty=100, min_size=5):
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
    changepoints = algo.predict(pen=penalty)
    return changepoints[:-1] if changepoints[-1] == len(signal) else changepoints

# ==========================================
# 5. CÁLCULO DE ESTADÍSTICAS DE SEGMENTOS
# ==========================================
def calculate_segment_stats(signal, changepoints):
    segments = [0] + list(changepoints) + [len(signal)]
    stats = []
    for i in range(len(segments) - 1):
        start_idx = segments[i]
        end_idx = segments[i+1]
        segment = signal[start_idx:end_idx]
        stats.append({
            'start_idx': start_idx, 'end_idx': end_idx,
            'mean': float(np.mean(segment)),
            'std': float(np.std(segment))
        })
    return stats

# ==========================================
# 6. EVALUACIÓN DE UMBRALES
# ==========================================
def evaluate_thresholds(segment_after_mean, dns_target):
    thresholds = THRESHOLDS[dns_target]
    return bool(segment_after_mean >= thresholds['warning']), bool(segment_after_mean >= thresholds['critical'])

# ==========================================
# 7. ALMACENAMIENTO DE CHANGEPOINTS EN BD
# ==========================================
def store_checkpoints_in_db(dns_target, changepoints, df, metric_name):
    signal = df[metric_name].values
    segment_stats = calculate_segment_stats(signal, changepoints)
    
    insert_query = """
        INSERT INTO link_health_changepoints 
        (dns_target, change_time, segment_before_mean, segment_before_std,
         segment_after_mean, segment_after_std, crosses_warning_threshold,
         crosses_critical_threshold, detected_at)
        VALUES 
        (:dns_target, :change_time, :before_mean, :before_std,
         :after_mean, :after_std, :crosses_warning, :crosses_critical, :detected_at)
    """
    
    checkpoints_stored = 0
    with engine.connect() as conn:
        for i, cp_idx in enumerate(changepoints):
            before_stats = segment_stats[i]
            after_stats = segment_stats[i + 1]
            crosses_warning, crosses_critical = evaluate_thresholds(after_stats['mean'], dns_target)
            change_time = df.index[cp_idx].to_pydatetime()
            
            conn.execute(text(insert_query), {
                'dns_target': dns_target.upper(),
                'change_time': change_time,
                'before_mean': float(before_stats['mean']),
                'before_std': float(before_stats['std']),
                'after_mean': float(after_stats['mean']),
                'after_std': float(after_stats['std']),
                'crosses_warning': bool(crosses_warning),
                'crosses_critical': bool(crosses_critical),
                'detected_at': datetime.now(timezone.utc)
            })
            checkpoints_stored += 1
            
            threshold_info = " 🔴 CRITICAL" if crosses_critical else (" 🟡 WARNING" if crosses_warning else "")
            print(f"⏱️ {change_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"{before_stats['mean']:.2f}ms (σ={before_stats['std']:.2f}) ➔ "
                  f"{after_stats['mean']:.2f}ms (σ={after_stats['std']:.2f}){threshold_info}")
        conn.commit()
    print(f"✅ {checkpoints_stored} changepoints de {dns_target.upper()} almacenados en BD")

# ==========================================
# 8. FEATURE ENGINEERING PARA ISOLATION FOREST
# ==========================================
def engineer_features_for_isolation_forest(df):
    df_ml = df.copy()
    df_ml['hour'] = df_ml.index.hour
    df_ml['day_of_week'] = df_ml.index.dayofweek
    df_ml['is_business_hours'] = df_ml['hour'].apply(lambda x: 1 if 8 <= x <= 18 else 0)
    
    window = 20
    for dns in ['dns1', 'dns2']:
        df_ml[f'{dns}_latency_rolling_mean'] = df_ml[f'{dns}_latency_ms'].rolling(window=window, min_periods=1).mean()
        df_ml[f'{dns}_latency_rolling_std'] = df_ml[f'{dns}_latency_ms'].rolling(window=window, min_periods=1).std().fillna(0)
        df_ml[f'{dns}_jitter_rolling_mean'] = df_ml[f'{dns}_jitter_ms'].rolling(window=window, min_periods=1).mean()
    return df_ml

# ==========================================
# 9. ALMACENAMIENTO DE DATASET ENRIQUECIDO EN BD (CORREGIDO)
# ==========================================
def store_enriched_dataset_in_db(df_ml):
    print("Almacenando dataset enriquecido en la hipertabla link_health_metrics_enriched...")
    try:
        # Solo intentamos hacer INSERT (append). La tabla YA debe existir (creada por el DBA).
        df_ml.to_sql('link_health_metrics_enriched', engine, if_exists='append', index=True, chunksize=5000)
        print(f"✅ {len(df_ml)} registros enriquecidos almacenados exitosamente en BD")
    except Exception as e:
        if "relation \"link_health_metrics_enriched\" does not exist" in str(e) or "permission denied" in str(e):
            print("❌ ERROR: La tabla 'link_health_metrics_enriched' no existe o no tienes permisos.")
            print("💡 SOLUCIÓN: Ejecuta el script SQL de creación de hipertabla proporcionado en la documentación como usuario 'postgres'.")
        else:
            print(f"❌ Error inesperado al guardar en BD: {e}")

# ==========================================
# 10. VISUALIZACIÓN
# ==========================================
def plot_changepoints(df, metric_name, changepoints, penalty, dns_target):
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[metric_name], label=f'{metric_name} (Raw)', color='blue', alpha=0.6)
    
    signal = df[metric_name].values
    segments = [0] + changepoints + [len(signal)]
    
    for i in range(len(segments) - 1):
        start_idx = segments[i]
        end_idx = segments[i+1]
        segment_mean = np.mean(signal[start_idx:end_idx])
        plt.hlines(segment_mean, df.index[start_idx], df.index[end_idx-1], 
                   colors='red', linestyles='dashed', linewidth=2, 
                   label='Segment Mean (Baseline)' if i==0 else "")
        
    for cp in changepoints:
        plt.axvline(df.index[cp], color='green', linestyle='-', linewidth=1.5, alpha=0.8)
    
    thresholds = THRESHOLDS[dns_target]
    plt.axhline(thresholds['warning'], color='orange', linestyle=':', linewidth=2, label=f"Warning ({thresholds['warning']}ms)", alpha=0.7)
    plt.axhline(thresholds['critical'], color='red', linestyle=':', linewidth=2, label=f"Critical ({thresholds['critical']}ms)", alpha=0.7)
        
    plt.title(f'Detección de Changepoints (PELT) en {metric_name}\nPenalidad: {penalty}', fontsize=14)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Latencia (ms)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    df = load_data_from_db()
    
    if df is not None:
        PENALTY = 100 
        
        print(f"\n🔍 Detectando changepoints en DNS1 con penalidad {PENALTY}...")
        cp_dns1 = detect_changepoints_pelt(df['dns1_latency_ms'].values, penalty=PENALTY, min_size=5)
        
        print(f"\n🔍 Detectando changepoints en DNS2 con penalidad {PENALTY}...")
        cp_dns2 = detect_changepoints_pelt(df['dns2_latency_ms'].values, penalty=PENALTY, min_size=5)
        
        print("\n--- 💾 CHANGEPOINTS DNS1 ---")
        store_checkpoints_in_db('dns1', cp_dns1, df, 'dns1_latency_ms')
        
        print("\n--- 📊 CHANGEPOINTS DNS2 ---")
        store_checkpoints_in_db('dns2', cp_dns2, df, 'dns2_latency_ms')
        
        # Descomenta las siguientes líneas si deseas ver los gráficos
        # plot_changepoints(df, 'dns1_latency_ms', cp_dns1, PENALTY, 'dns1')
        # plot_changepoints(df, 'dns2_latency_ms', cp_dns2, PENALTY, 'dns2')
        
        print("\n⚙️ Generando features para Isolation Forest...")
        df_ml_ready = engineer_features_for_isolation_forest(df)
        
        store_enriched_dataset_in_db(df_ml_ready)
        
        print("\n✅ Proceso completado exitosamente.")
