import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# ==========================================
# 1. CONFIGURACIÓN DE CONEXIÓN
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
# 2. CARGA DE DATOS AVANZADOS (LINK_HEALTH_FEATURES)
# ==========================================
def load_advanced_features():
    # Consultamos la tabla con las características avanzadas que proporcionaste
    query = """
        SELECT 
            time, cycle_number, dns1_latency_ms, dns2_latency_ms,
            z_score_dns1, z_score_dns2, 
            latency_velocity_dns1, latency_velocity_dns2,
            latency_trend_15min_dns1, latency_trend_15min_dns2,
            cv_dns1, cv_dns2,
            hour_of_day, day_of_week
        FROM link_health_features 
        ORDER BY time ASC;
    """
    print("Cargando características avanzadas desde TimescaleDB...")
    
    # Cargamos TODO el dataset para evaluación posterior
    df_full = pd.read_sql(query, con=engine)
    df_full['time'] = pd.to_datetime(df_full['time'])
    df_full.set_index('time', inplace=True)
    
    # FILTRO CRÍTICO (Opción A): Entrenar SOLO con el estado saludable de la red
    # Esto fuerza al modelo a aprender que z_score≈0 y velocity≈0 es lo "normal"
    df_train = df_full[(df_full['dns1_latency_ms'] < 5.0) & (df_full['dns2_latency_ms'] < 5.0)].copy()
    
    print(f"📊 Total de registros en BD: {len(df_full)}")
    print(f"✅ Registros 'normales' (< 5ms) para entrenamiento: {len(df_train)} ({len(df_train)/len(df_full)*100:.1f}%)")
    
    return df_train, df_full

# ==========================================
# 3. ENTRENAMIENTO DEL ISOLATION FOREST
# ==========================================
def train_model(df_train, df_full):
    # Usamos las características avanzadas que capturan DESVIACIÓN y TENDENCIA
    feature_cols = [
        'hour_of_day', 'day_of_week',
        'z_score_dns1', 'z_score_dns2',
        'latency_velocity_dns1', 'latency_velocity_dns2',
        'latency_trend_15min_dns1', 'latency_trend_15min_dns2',
        'cv_dns1', 'cv_dns2'
    ]
    
    # Limpiamos valores nulos (fillna(0) es seguro para z-scores y velocidades en estado estable)
    X_train = df_train[feature_cols].fillna(0).dropna()
    
    print("Entrenando Isolation Forest SOLO con línea base normal (Opción A)...")
    # Contamination muy bajo porque el set de entrenamiento está limpio
    model = IsolationForest(n_estimators=150, contamination=0.01, random_state=42, n_jobs=-1)
    model.fit(X_train)
    
    print("Calculando scores de anomalía para TODOS los datos (incluyendo degradados)...")
    X_full = df_full[feature_cols].fillna(0)
    
    # Invertimos el signo: score ALTO = MÁS anómalo (ideal para alertas en Grafana)
    df_full['anomaly_score'] = -model.decision_function(X_full)
    df_full['is_anomaly'] = model.predict(X_full) # 1 = Normal, -1 = Anomalía
    
    joblib.dump(model, 'isolation_forest_bgp_advanced.pkl')
    print("✅ Modelo avanzado entrenado y guardado como 'isolation_forest_bgp_advanced.pkl'")
    
    return df_full, model

# ==========================================
# 4. VALIDACIÓN CONTRA CHANGEPOINTS
# ==========================================
def validate_against_changepoints(df_ml):
    print("\n🔍 Validando modelo contra Changepoints detectados (Ground Truth)...")
    
    query_cp = """
        SELECT dns_target, change_time, crosses_warning_threshold, crosses_critical_threshold 
        FROM link_health_changepoints 
        ORDER BY change_time ASC;
    """
    df_cp = pd.read_sql(query_cp, con=engine)
    df_cp['change_time'] = pd.to_datetime(df_cp['change_time'])
    
    validation_results = []
    
    for _, cp in df_cp.iterrows():
        target = cp['dns_target'].lower()
        cp_time = cp['change_time']
        
        # Ventana de validación: 3 min antes y 5 min después del changepoint
        window_start = cp_time - timedelta(minutes=3)
        window_end = cp_time + timedelta(minutes=5)
        
        mask = (df_ml.index >= window_start) & (df_ml.index <= window_end)
        window_data = df_ml[mask]
        
        if len(window_data) > 0:
            avg_score = window_data['anomaly_score'].mean()
            anomaly_count = (window_data['is_anomaly'] == -1).sum()
            
            # Consideramos "DETECTADO" si al menos el 30% de la ventana o >= 2 puntos son anómalos
            status = "✅ DETECTADO" if anomaly_count >= 2 else "⚠️ NO DETECTADO"
            
            validation_results.append({
                'change_time': cp_time,
                'dns_target': target,
                'threshold_crossed': 'CRITICAL' if cp['crosses_critical_threshold'] else ('WARNING' if cp['crosses_warning_threshold'] else 'NORMAL'),
                'avg_anomaly_score': round(avg_score, 4),
                'anomalies_in_window': int(anomaly_count),
                'validation_status': status
            })
            
    val_df = pd.DataFrame(validation_results)
    if not val_df.empty:
        print("\n--- 📊 RESULTADO DE VALIDACIÓN DEL MODELO ---")
        print(val_df[['change_time', 'dns_target', 'threshold_crossed', 'avg_anomaly_score', 'validation_status']].to_string(index=False))
        
        success_rate = (val_df['validation_status'] == '✅ DETECTADO').mean() * 100
        print(f"\n🎯 Tasa de detección de anomalías en ventanas de Changepoint: {success_rate:.1f}%")
        
        if success_rate < 50:
            print("💡 SUGERENCIA: La tasa es baja. Verifica que la tabla 'link_health_features' tenga valores calculados correctamente (no todos NULL o 0).")

# ==========================================
# 5. ALMACENAMIENTO DE RESULTADOS EN BD
# ==========================================
def save_ml_results_to_db(df_ml):
    print("\n💾 Almacenando resultados de ML en TimescaleDB...")
    
    df_to_save = df_ml[['dns1_latency_ms', 'dns2_latency_ms', 'anomaly_score', 'is_anomaly']].copy()
    
    try:
        # Asumimos que la tabla link_health_metrics_ml_results ya fue creada por el script SQL anterior
        df_to_save.to_sql('link_health_metrics_ml_results', engine, if_exists='append', index=True, chunksize=5000)
        print(f"✅ {len(df_to_save)} registros con scores de anomalía almacenados exitosamente.")
    except Exception as e:
        print(f"❌ Error al guardar en BD. Asegúrate de haber ejecutado el script SQL de creación de la hipertabla.")
        print(f"Detalle: {e}")

# ==========================================
# 6. VISUALIZACIÓN DE EJEMPLO
# ==========================================
def plot_anomaly_detection(df_ml, dns_target='dns1'):
    plt.figure(figsize=(15, 6))
    plt.plot(df_ml.index, df_ml[f'{dns_target}_latency_ms'], label=f'{dns_target} Latency', color='gray', alpha=0.5)
    
    anomalies = df_ml[df_ml['is_anomaly'] == -1]
    plt.scatter(anomalies.index, anomalies[f'{dns_target}_latency_ms'], 
                color='red', label='ML Anomaly', s=15, zorder=5)
    
    plt.title(f'Detección de Anomalías (Features Avanzadas) - {dns_target.upper()}', fontsize=14)
    plt.xlabel('Tiempo')
    plt.ylabel('Latencia (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Cargar datos avanzados
    df_train, df_full = load_advanced_features()
    
    if df_train is not None and len(df_train) > 100:
        # 2. Entrenar solo con lo limpio, evaluar sobre todo
        df_with_scores, trained_model = train_model(df_train, df_full)
        
        # 3. Validar contra la verdad terrena
        validate_against_changepoints(df_with_scores)
        
        # 4. Guardar en BD para Grafana
        save_ml_results_to_db(df_with_scores)
        
        # 5. Mostrar gráfico
        print("\n📈 Generando gráfico de validación visual...")
        plot_anomaly_detection(df_with_scores, dns_target='dns1')
        plot_anomaly_detection(df_with_scores, dns_target='dns2')
        
        print("\n✅ Pipeline de entrenamiento avanzado completado exitosamente.")
    else:
        print("❌ No hay suficientes datos 'normales' (< 5ms) para entrenar el modelo.")
