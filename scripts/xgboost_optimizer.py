#!/usr/bin/env python3
"""
xgboost_optimizer.py — v3: MULTICLASE sobre ml_features real (Opción A)
═══════════════════════════════════════════════════════════════════════════════
✅ REESCRITO COMPLETO respecto a la versión anterior (binaria, tabla legacy):

CAMBIO DE FONDO — binario → multiclase:
├─ objective: 'binary:logistic' → 'multi:softprob' (num_class=4)
├─ target: should_failover/failover_event → target_decision
│   (normal / degradacion / failover / retorno — Opción A: un solo modelo)
├─ scale_pos_weight (solo válido para 2 clases) → sample_weight balanceado
│   por clase (sklearn.utils.class_weight.compute_sample_weight)
├─ Métricas: precision/recall/f1 → promediado MACRO (no penalizar clases
│   minoritarias); ROC-AUC → One-vs-Rest promediado (ver slide 6 actualizada)
└─ eval_metric nativo XGBoost: ['mlogloss', 'merror'] en vez de
   ['error','aucpr','auc','logloss'] (aucpr/auc binarios no aplican acá)

CAMBIO DE FONDO — features desde ml_features real (no la tabla legacy):
├─ Elimina TODAS las columnas ya dropeadas en la migración v2 de ml_features
│   (latency_ratio, quality_index, score_difference, margin_exceeds_threshold,
│   absolute_severity, relative_diff_ms, relative_severity, combined_severity,
│   is_combined_anomaly, rolling_mean/std/p95, z_score_severity — no existen)
├─ Incorpora el set real: DNS1/DNS2 separados, componentes normalizados
│   Etapa 1, features temporales/estadísticas Etapa 2 (z-score, CV, p95_dev,
│   z_deriv, trend/velocity/aceleración por métrica), contextuales, y
│   contexto de cambios de provider (ya sin fuga temporal, ver v2.1 del
│   feature_engine)
└─ ⚠️ EXCLUYE explícitamente 'degradation_cycle' y 'provider_changed':
    - degradation_cycle es el contador que la propia regla del motor usa
      para decidir failover — incluirlo sería dejar que el modelo aprenda
      a leer el bookkeeping de la regla existente, no las métricas de red.
    - provider_changed es casi un espejo binario del target (True
      exactamente cuando target_decision es 'failover' o 'retorno') — esto
      NO estaba excluido en la versión anterior y es una fuga real.
   'quality_status' y 'max_score'/'score_dns1'/'score_dns2' SÍ se incluyen:
   son medidas (aunque derivadas) del estado de la red en ese ciclo, no
   contadores del proceso de decisión — es justo lo que queremos que el
   modelo evalúe (¿el score compuesto actual explica bien la decisión, o
   las métricas individuales explican más?).

SE CONSERVA (metodología válida, independiente del binario/multiclase):
  - _to_dmatrix() con sample_weight              (adaptado a multiclase)
  - find_optimal_rounds() vía xgb.cv()           (adaptado a mlogloss)
  - analyze_learning_curve()                     (adaptado a mlogloss/merror)
  - tune_hyperparameters() vía Optuna (BO)       (adaptado a mlogloss)
  - analyze_feature_stability()                  (sin cambios de fondo)

⚠️ NOTA DE ALCANCE (dataset chico): con ~15-20 filas por corrida de prueba,
el tuning bayesiano de 7 hiperparámetros puede sobreajustar su propia
búsqueda. n_trials tiene un default conservador (15, no 30) y todo el
pipeline degrada con gracia (reduce folds, avisa) en vez de fallar. Escalar
n_trials cuando el dataset crezca (ver pendiente: captura de más días /
generador de escenarios sintéticos).

FLUJO RECOMENDADO:
  optimizer = ScoringWeightOptimizer()
  best_params = optimizer.tune_hyperparameters(df, n_trials=15)
  optimizer.train_with_cv(df, best_params=best_params)
  weights = optimizer.get_optimized_weights()
═══════════════════════════════════════════════════════════════════════════════
"""

import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Clases del target multiclase (Opción A) — orden fijo para reproducibilidad
DECISION_CLASSES = ['normal', 'degradacion', 'failover', 'retorno']

# ✅ Features EXCLUIDAS explícitamente por fuga de datos (ver docstring arriba)
LEAKAGE_FEATURES = {'degradation_cycle', 'provider_changed'}

# ✅ Identificadores / no-features (no aportan patrón generalizable)
IDENTIFIER_COLUMNS = {'time', 'cycle_number', 'target_decision', 'decision'}

# ✅ Columnas categóricas que requieren encoding antes de entrenar
CATEGORICAL_FEATURES = {
    'provider': {'PROVIDER1': 0, 'PROVIDER2': 1},
    'quality_status': {'excellent': 0, 'warning': 1, 'critical': 2},
}

# ✅ Columnas booleanas que XGBoost necesita como int
BOOLEAN_FEATURES = ['is_business_hours', 'is_peak_traffic', 'is_weekend', 'loss_spike_dns1', 'loss_spike_dns2']

# ✅ Taxonomía de features (ver slide "Taxonomía de features derivadas") —
# usada en get_optimized_weights() para agrupar importancia por familia.
FEATURE_GROUPS = {
    'peer': [
        'peer_latency_ms', 'peer_jitter_ms', 'peer_loss_pct', 'peer_norm',
        'z_score_peer', 'cv_peer', 'p95_dev_peer', 'z_deriv_peer',
        'latency_trend_5min_peer', 'latency_trend_15min_peer',
        'latency_velocity_peer', 'latency_acceleration_peer',
    ],
    'dns': [
        'dns1_latency_ms', 'dns1_jitter_ms', 'dns1_loss_pct',
        'dns2_latency_ms', 'dns2_jitter_ms', 'dns2_loss_pct',
        'dns1_norm', 'dns2_norm',
        'z_score_dns1', 'z_score_dns2', 'cv_dns1', 'cv_dns2',
        'p95_dev_dns1', 'p95_dev_dns2', 'z_deriv_dns1', 'z_deriv_dns2',
        'latency_trend_5min_dns1', 'latency_trend_5min_dns2',
        'latency_trend_15min_dns1', 'latency_trend_15min_dns2',
        'latency_velocity_dns1', 'latency_velocity_dns2',
        'latency_acceleration_dns1', 'latency_acceleration_dns2',
    ],
    'jitter': ['jitter1_norm', 'jitter2_norm', 'z_score_jitter1', 'z_score_jitter2'],
    'loss': ['loss1_norm', 'loss2_norm', 'z_score_loss1', 'z_score_loss2',
             'loss_spike_dns1', 'loss_spike_dns2',
             'severity_multiplier_dns1', 'severity_multiplier_dns2'],
    'score_compuesto': ['base_score_dns1', 'base_score_dns2', 'score_dns1',
                         'score_dns2', 'max_score', 'quality_status'],
    'contextual': ['hour_of_day', 'day_of_week', 'is_business_hours',
                   'is_peak_traffic', 'is_weekend'],
    'contexto_cambios': ['provider_changes_last_hour', 'time_since_last_change_min'],
    'provider': ['provider'],
}


class ScoringWeightOptimizer:
    """
    XGBoost multiclase + Cross-Validation para analizar qué features de
    ml_features explican mejor target_decision (normal/degradacion/
    failover/retorno), como insumo para re-calibrar la fórmula de scoring
    (Opción B: este ranking + Logistic Regression para coeficientes).
    """

    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.feature_importance_std = None
        self.features_used = None
        self.cv_scores = None
        self.cv_importances = None
        self.target_column = 'target_decision'
        self.label_encoder = LabelEncoder().fit(DECISION_CLASSES)
        self.num_class = len(DECISION_CLASSES)

        self.optimal_rounds = None
        self.learning_curve_data = None
        self.best_params = None
        self.tuning_study = None

    # ════════════════════════════════════════════════════════════════════
    # DMatrix con sample_weight balanceado por clase (reemplaza scale_pos_weight)
    # ════════════════════════════════════════════════════════════════════
    def _to_dmatrix(self, X, y_encoded):
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_encoded)
        return xgb.DMatrix(
            X_vals,
            label=y_encoded,
            weight=sample_weights,
            feature_names=list(X.columns) if hasattr(X, 'columns') else None
        )

    # ════════════════════════════════════════════════════════════════════
    # Búsqueda de rondas óptimas vía xgb.cv() nativo (multiclase)
    # ════════════════════════════════════════════════════════════════════
    def find_optimal_rounds(self, X, y_encoded, params, max_rounds=500,
                             early_stop=15, n_folds=5, random_state=42):
        logger.info(f"\n🔍 Buscando rondas óptimas (max={max_rounds}, early_stop={early_stop})...")

        dtrain = self._to_dmatrix(X, y_encoded)
        params_with_metrics = {
            **params,
            'objective': 'multi:softprob',
            'num_class': self.num_class,
            'eval_metric': ['mlogloss', 'merror'],
        }

        n_folds_safe = min(n_folds, min(np.bincount(y_encoded)))
        if n_folds_safe < n_folds:
            logger.warning(f"⚠️ Clase minoritaria tiene <{n_folds} muestras → n_folds={n_folds_safe}")
        n_folds_safe = max(2, n_folds_safe)

        cv_results = xgb.cv(
            params=params_with_metrics,
            dtrain=dtrain,
            num_boost_round=max_rounds,
            nfold=n_folds_safe,
            stratified=True,
            early_stopping_rounds=early_stop,
            verbose_eval=False,
            seed=random_state,
        )

        optimal_rounds = len(cv_results)
        opt_idx = cv_results['test-mlogloss-mean'].idxmin()

        self.learning_curve_data = cv_results
        self.optimal_rounds = optimal_rounds

        logger.info(f"   ✅ Rondas óptimas: {optimal_rounds}")
        logger.info(f"   ✅ test-mlogloss:  {cv_results['test-mlogloss-mean'].iloc[opt_idx]:.6f} "
                    f"± {cv_results['test-mlogloss-std'].iloc[opt_idx]:.6f}")
        if 'test-merror-mean' in cv_results.columns:
            logger.info(f"   ✅ test-merror:    {cv_results['test-merror-mean'].iloc[opt_idx]:.6f}")

        return optimal_rounds, cv_results

    # ════════════════════════════════════════════════════════════════════
    # Curva de aprendizaje (multiclase: mlogloss/merror en vez de logloss/auc)
    # ════════════════════════════════════════════════════════════════════
    def analyze_learning_curve(self, cv_results=None):
        if cv_results is None:
            cv_results = self.learning_curve_data
        if cv_results is None:
            logger.error("❌ Ejecutar find_optimal_rounds() primero")
            return {}

        train_ll = cv_results['train-mlogloss-mean']
        test_ll = cv_results['test-mlogloss-mean']
        gap = test_ll - train_ll

        opt_idx = test_ll.idxmin()
        best_round = int(opt_idx) + 1

        gap_delta = gap.diff().rolling(5).mean()
        overfit_candidates = gap_delta[gap_delta > 0.0001].index
        overfit_round = int(overfit_candidates[0]) + 1 if len(overfit_candidates) else None

        summary = {
            'best_round': best_round,
            'overfit_start_round': overfit_round,
            'test_mlogloss_at_best': float(test_ll.iloc[opt_idx]),
            'test_mlogloss_std': float(cv_results['test-mlogloss-std'].iloc[opt_idx]),
            'gap_train_test': float(gap.iloc[opt_idx]),
            'total_rounds': len(cv_results),
        }
        if 'test-merror-mean' in cv_results.columns:
            summary['test_merror_at_best'] = float(cv_results['test-merror-mean'].iloc[opt_idx])

        logger.info(f"\n📈 Learning Curve Analysis:")
        logger.info(f"   Ronda óptima (min test-mlogloss): {best_round}")
        logger.info(f"   Inicio overfitting detectado:     {overfit_round or 'no detectado'}")
        logger.info(f"   Gap train/test en ronda óptima:   {summary['gap_train_test']:.6f}")

        return summary

    # ════════════════════════════════════════════════════════════════════
    # Bayesian Optimization (Optuna) — adaptado a multiclase
    # ════════════════════════════════════════════════════════════════════
    def tune_hyperparameters(self, df, n_trials=15, random_state=42):
        logger.info("\n" + "═" * 80)
        logger.info(f"🔍 Hyperparameter Tuning — Bayesian Optimization (Optuna, n_trials={n_trials})")
        logger.info("═" * 80)

        X, y_encoded, _ = self.prepare_features(df)
        dtrain = self._to_dmatrix(X, y_encoded)
        n_folds_safe = max(2, min(5, min(np.bincount(y_encoded))))

        trial_results = []

        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': self.num_class,
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'eta': trial.suggest_float('eta', 0.01, 0.30, log=True),
                'subsample': trial.suggest_float('subsample', 0.60, 1.00),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 1.00),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'eval_metric': ['mlogloss', 'merror'],
            }

            cv = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=300,
                nfold=n_folds_safe,
                stratified=True,
                early_stopping_rounds=15,
                verbose_eval=False,
                seed=random_state,
            )

            best_mlogloss = float(cv['test-mlogloss-mean'].min())
            trial.set_user_attr('n_estimators', len(cv))
            trial_results.append({'trial': trial.number + 1, 'mlogloss': best_mlogloss})

            if len(trial_results) % 5 == 0:
                best_so_far = min(r['mlogloss'] for r in trial_results)
                logger.info(f"   Trial {len(trial_results):>3}/{n_trials} | "
                            f"mlogloss={best_mlogloss:.6f} | best_so_far={best_so_far:.6f}")

            return best_mlogloss

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_params['n_estimators'] = best_trial.user_attrs.get('n_estimators', 100)

        logger.info(f"\n   ✅ Optimización completada en {n_trials} trials")
        logger.info(f"   Best test-mlogloss: {best_trial.value:.6f}")
        logger.info(f"   Parámetros óptimos:")
        for p, val in best_trial.params.items():
            fmt_val = f'{val:.4f}' if isinstance(val, float) else str(val)
            logger.info(f"     {p:<22} {fmt_val}")
        logger.info(f"     {'n_estimators':<22} {best_params['n_estimators']}")

        self.best_params = best_params
        self.tuning_study = study
        return best_params

    # ════════════════════════════════════════════════════════════════════
    # prepare_features — REESCRITO para el schema real de ml_features
    # ════════════════════════════════════════════════════════════════════
    def prepare_features(self, df):
        """
        Prepara X (features) e y (target_decision codificado) desde un
        DataFrame de ml_features real.

        Excluye explícitamente LEAKAGE_FEATURES (degradation_cycle,
        provider_changed) e IDENTIFIER_COLUMNS (time, cycle_number).
        Codifica 'provider' y 'quality_status' (categóricas) y castea
        booleanas a int.
        """
        logger.info("\n🔧 Preparando features desde ml_features...")

        if 'target_decision' not in df.columns:
            raise ValueError("❌ Falta la columna target_decision en el DataFrame")

        # Filtrar filas sin target válido (no deberían existir si
        # feature_engine_incremental.py ya excluyó error/failover_inmediato,
        # pero se valida por robustez)
        df = df[df['target_decision'].isin(DECISION_CLASSES)].copy()
        if df.empty:
            raise ValueError("❌ No hay filas con target_decision válido "
                              f"(esperado uno de {DECISION_CLASSES})")

        exclude = LEAKAGE_FEATURES | IDENTIFIER_COLUMNS
        candidate_cols = [c for c in df.columns if c not in exclude]

        excluded_present = [c for c in df.columns if c in LEAKAGE_FEATURES]
        if excluded_present:
            logger.info(f"   🚫 Excluidas por fuga de datos: {excluded_present}")

        X = df[candidate_cols].copy()

        # Encoding de categóricas
        for col, mapping in CATEGORICAL_FEATURES.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(-1).astype(int)

        # Booleanas -> int
        for col in BOOLEAN_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(bool).astype(int)
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)

        # NULLs -> 0 (loss/z-score con varianza cero quedan NULL por diseño,
        # ver feature_engine_incremental.py — 0 es una imputación razonable:
        # "sin desviación detectable" cuando no hay suficiente historia o
        # varianza)
        n_nulls = X.isnull().sum().sum()
        if n_nulls > 0:
            logger.warning(f"⚠️ {n_nulls} valores NULL detectados — imputando con 0")
            X = X.fillna(0)

        # ⚠️ FIX: fillna(0) reemplaza los valores pero NO cambia el dtype de
        # la columna. Con conexión psycopg2 "cruda" (no SQLAlchemy), una
        # columna con muchos/todos NULL (ej. z_score_loss1/2 cuando nunca
        # hubo pérdida simulada, o time_since_last_change_min antes del
        # primer cambio de provider) llega como dtype=object con None de
        # Python adentro — el mismo problema de tipos que ya resolvimos para
        # la columna 'time' en feature_engine_incremental.py. fillna(0) deja
        # el VALOR en 0 pero el dtype sigue siendo 'object', y XGBoost
        # rechaza cualquier columna que no sea int/float/bool. Se fuerza la
        # conversión numérica explícita acá, columna por columna.
        object_cols = X.select_dtypes(include='object').columns.tolist()
        if object_cols:
            logger.warning(f"⚠️ Forzando conversión numérica en columnas object: {object_cols}")
            for col in object_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        y_encoded = self.label_encoder.transform(df['target_decision'])
        self.features_used = list(X.columns)

        logger.info(f"📊 Features preparadas: {len(self.features_used)}")
        logger.info(f"📊 Distribución de clases:")
        for i, cls in enumerate(self.label_encoder.classes_):
            count = (y_encoded == i).sum()
            logger.info(f"   {cls:<15}: {count:4d} ({count/len(y_encoded)*100:5.1f}%)")

        return X, y_encoded, self.features_used

    # ════════════════════════════════════════════════════════════════════
    # train_with_cv — Stratified K-Fold, métricas macro + ROC-AUC OvR
    # ════════════════════════════════════════════════════════════════════
    def train_with_cv(self, df, n_splits=5, random_state=42, best_params=None):
        logger.info("\n" + "═" * 80)
        logger.info("🤖 XGBoost multiclase: Cross-Validation")
        logger.info("═" * 80)

        X, y_encoded, features = self.prepare_features(df)

        # ── Resolver hiperparámetros ──────────────────────────────────────
        if best_params is not None:
            hp = best_params
            logger.info("\n✅ Usando hiperparámetros de tune_hyperparameters()")
        else:
            logger.info("\n⚠️ best_params=None → usando defaults con find_optimal_rounds()")
            default_params = {
                'max_depth': 3, 'eta': 0.05, 'subsample': 0.8,
                'colsample_bytree': 0.7, 'reg_alpha': 0.1,
                'reg_lambda': 1.0, 'min_child_weight': 1,
            }
            optimal_n, _ = self.find_optimal_rounds(
                X, y_encoded, default_params,
                max_rounds=500, early_stop=15, n_folds=n_splits,
                random_state=random_state
            )
            hp = {**default_params, 'n_estimators': optimal_n}

        n_estimators = hp.get('n_estimators', self.optimal_rounds or 100)
        max_depth = hp.get('max_depth', 3)
        learning_rate = hp.get('eta', hp.get('learning_rate', 0.05))
        subsample = hp.get('subsample', 0.8)
        colsample_bytree = hp.get('colsample_bytree', 0.7)
        reg_alpha = hp.get('reg_alpha', 0.1)
        reg_lambda = hp.get('reg_lambda', 1.0)
        min_child_weight = hp.get('min_child_weight', 1)

        logger.info(f"\n   Hiperparámetros a usar:")
        logger.info(f"   n_estimators     = {n_estimators}")
        logger.info(f"   max_depth        = {max_depth}")
        logger.info(f"   learning_rate    = {learning_rate:.4f}")
        logger.info(f"   subsample        = {subsample:.2f}")
        logger.info(f"   colsample_bytree = {colsample_bytree:.2f}")
        logger.info(f"   reg_alpha        = {reg_alpha:.4f}")
        logger.info(f"   reg_lambda       = {reg_lambda:.4f}")
        logger.info(f"   min_child_weight = {min_child_weight}")

        # ── Validar folds contra la clase minoritaria ──────────────────────
        class_counts = np.bincount(y_encoded, minlength=self.num_class)
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < n_splits:
            logger.warning(f"⚠️ Clase minoritaria tiene {min_class_count} muestras "
                           f"para {n_splits} folds → reduciendo a {max(2, min_class_count)}")
            n_splits = max(2, min_class_count)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_metrics = {
            'precision_macro': [], 'recall_macro': [], 'f1_macro': [], 'roc_auc_ovr': [],
        }
        fold_importances = {feat: [] for feat in features}

        logger.info(f"\n🔄 {n_splits}-Fold Stratified CV (multiclase, promediado macro)...")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=self.num_class,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                eval_metric='mlogloss',
                verbosity=0,
            )
            model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            fold_metrics['precision_macro'].append(
                precision_score(y_test, y_pred, average='macro', zero_division=0))
            fold_metrics['recall_macro'].append(
                recall_score(y_test, y_pred, average='macro', zero_division=0))
            fold_metrics['f1_macro'].append(
                f1_score(y_test, y_pred, average='macro', zero_division=0))

            try:
                # ROC-AUC OvR requiere que TODAS las clases estén presentes en y_test
                present_classes = np.unique(y_test)
                if len(present_classes) >= 2:
                    fold_metrics['roc_auc_ovr'].append(
                        roc_auc_score(y_test, y_pred_proba, multi_class='ovr',
                                       average='macro', labels=list(range(self.num_class)))
                    )
                else:
                    fold_metrics['roc_auc_ovr'].append(np.nan)
            except Exception as e:
                logger.warning(f"   ⚠️ ROC-AUC no calculable en fold {fold}: {e}")
                fold_metrics['roc_auc_ovr'].append(np.nan)

            for feat, imp in zip(features, model.feature_importances_):
                fold_importances[feat].append(imp)

            logger.info(
                f"   Fold {fold}/{n_splits}: "
                f"Precision={fold_metrics['precision_macro'][-1]:.3f} | "
                f"Recall={fold_metrics['recall_macro'][-1]:.3f} | "
                f"F1={fold_metrics['f1_macro'][-1]:.3f} | "
                f"ROC-AUC(OvR)={fold_metrics['roc_auc_ovr'][-1]:.3f}"
            )

        self.cv_scores = fold_metrics
        self.cv_importances = fold_importances

        logger.info(f"\n📊 Cross-Validation Results ({n_splits} folds, macro-average):")
        logger.info("-" * 80)
        for metric, values in fold_metrics.items():
            valid = [v for v in values if not np.isnan(v)]
            if valid:
                logger.info(f"   {metric:<18}: {np.mean(valid):.4f} ± {np.std(valid):.4f}")

        avg_importances = {f: float(np.mean(fold_importances[f])) for f in features}
        std_importances = {f: float(np.std(fold_importances[f])) for f in features}

        self.feature_importance = pd.DataFrame({
            'feature': features,
            'importance': [avg_importances[f] for f in features],
            'importance_std': [std_importances[f] for f in features],
        }).sort_values('importance', ascending=False)
        self.feature_importance_std = self.feature_importance['importance_std'].values

        logger.info(f"\n🔄 Entrenando modelo final ({n_estimators} rondas, todos los datos)...")
        full_sample_weight = compute_sample_weight(class_weight='balanced', y=y_encoded)
        self.model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=self.num_class,
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            random_state=random_state, eval_metric='mlogloss', verbosity=0,
        )
        self.model.fit(X, y_encoded, sample_weight=full_sample_weight, verbose=False)

        self.analyze_feature_stability()
        if self.learning_curve_data is not None:
            self.analyze_learning_curve()

        return self.feature_importance

    # ════════════════════════════════════════════════════════════════════
    # Estabilidad de features entre folds — sin cambios de fondo
    # ════════════════════════════════════════════════════════════════════
    def analyze_feature_stability(self):
        logger.info(f"\n📊 Análisis de Estabilidad de Features:")
        logger.info("-" * 80)

        stability_data = []
        for _, row in self.feature_importance.iterrows():
            feat, mean_imp, std_imp = row['feature'], row['importance'], row['importance_std']
            cv = std_imp / mean_imp if mean_imp > 0.001 else 0.0
            stability = ("✅ ESTABLE" if cv < 0.3 else
                         "⚠️ MODERADA" if cv < 0.7 else "❌ INESTABLE")
            stability_data.append({'feature': feat, 'mean': mean_imp,
                                    'std': std_imp, 'cv': cv, 'stability': stability})

        stability_df = pd.DataFrame(stability_data).sort_values('mean', ascending=False)

        for _, row in stability_df.head(15).iterrows():
            bar = "█" * int(row['mean'] * 100 / 2)
            logger.info(f"   {row['feature']:28s} {bar:25s} "
                        f"{row['mean']*100:5.1f}% ± {row['std']*100:4.1f}% "
                        f"(CV={row['cv']:.2f}) {row['stability']}")

        stable = sum(1 for d in stability_data if d['cv'] < 0.3)
        moderate = sum(1 for d in stability_data if 0.3 <= d['cv'] < 0.7)
        unstable = sum(1 for d in stability_data if d['cv'] >= 0.7)
        logger.info(f"\n   ✅ Estables: {stable:2d} | ⚠️ Moderadas: {moderate:2d} | "
                    f"❌ Inestables: {unstable:2d}")

        return stability_df

    # ════════════════════════════════════════════════════════════════════
    # get_optimized_weights — agrupado por la taxonomía REAL (peer/dns/
    # jitter/loss/score_compuesto/contextual/contexto_cambios/provider)
    # ════════════════════════════════════════════════════════════════════
    def get_optimized_weights(self):
        if self.feature_importance is None:
            logger.error("❌ Entrenar el modelo primero")
            return None

        logger.info("\n" + "═" * 80)
        logger.info("📈 PESOS OPTIMIZADOS (Promedio Cross-Validation, multiclase)")
        logger.info("═" * 80)

        imp_map = dict(zip(self.feature_importance['feature'], self.feature_importance['importance']))

        logger.info("\n🔍 Top 20 features:")
        for idx, (_, row) in enumerate(self.feature_importance.head(20).iterrows(), 1):
            pct = row['importance'] * 100
            bar = "█" * int(pct / 2)
            logger.info(f"   {idx:2d}. {row['feature']:28s} {bar:25s} "
                        f"{pct:5.1f}% ± {row['importance_std']*100:4.1f}%")

        group_totals = {}
        for group, cols in FEATURE_GROUPS.items():
            group_totals[group] = float(sum(imp_map.get(c, 0.0) for c in cols))

        logger.info("\n📋 RESUMEN POR FAMILIA (taxonomía ml_features):")
        for group, total in sorted(group_totals.items(), key=lambda x: -x[1]):
            pct = total * 100
            bar = "█" * int(pct / 2)
            logger.info(f"   {group:20s} {bar:25s} {pct:5.1f}%")

        # ── Traducción a pesos candidatos comparables con la fórmula actual ──
        # (peer=0.27 / dns=0.50 / jitter=0.23 — loss entra como multiplicador,
        # no como peso, así que se reporta aparte)
        core = group_totals['peer'] + group_totals['dns'] + group_totals['jitter']
        candidate_peer_w = group_totals['peer'] / core if core > 0 else 0.27
        candidate_dns_w = group_totals['dns'] / core if core > 0 else 0.50
        candidate_jitter_w = group_totals['jitter'] / core if core > 0 else 0.23

        logger.info(f"\n🎯 Pesos candidatos (renormalizados peer+dns+jitter=1, comparables a la fórmula actual):")
        logger.info(f"   peer   = {candidate_peer_w:.4f}  (actual: 0.27)")
        logger.info(f"   dns    = {candidate_dns_w:.4f}  (actual: 0.50)")
        logger.info(f"   jitter = {candidate_jitter_w:.4f}  (actual: 0.23)")
        logger.info(f"   ⚠️ Esto es solo RANKING (Gain) — para coeficientes lineales")
        logger.info(f"      directamente interpretables, ver el paso de Logistic Regression.")

        if self.cv_scores:
            logger.info(f"\n📊 Cross-Validation Summary (macro-average):")
            logger.info("-" * 80)
            for metric, values in self.cv_scores.items():
                valid = [v for v in values if not np.isnan(v)]
                if valid:
                    logger.info(f"   {metric:<18}: {np.mean(valid):.4f} ± {np.std(valid):.4f}")

        if self.best_params:
            logger.info(f"\n⚙️ Hiperparámetros óptimos (Bayesian Optimization):")
            for k, v in self.best_params.items():
                fmt = f'{v:.4f}' if isinstance(v, float) else str(v)
                logger.info(f"   {k:<22}: {fmt}")

        return {
            'group_importances': group_totals,
            'candidate_weights': {
                'peer': float(candidate_peer_w),
                'dns': float(candidate_dns_w),
                'jitter': float(candidate_jitter_w),
            },
            'current_weights': {'peer': 0.27, 'dns': 0.50, 'jitter': 0.23},
            'all_importances': self.feature_importance.to_dict('list'),
            'cv_scores': self.cv_scores,
            'target_column': self.target_column,
            'decision_classes': list(self.label_encoder.classes_),
            'optimal_rounds': self.optimal_rounds,
            'best_hyperparams': self.best_params,
            'learning_curve_summary': (self.analyze_learning_curve()
                                        if self.learning_curve_data is not None else None),
            'recommendations': {
                'temporal_features_matter': group_totals.get('dns', 0) > 0.10 and
                    any(imp_map.get(c, 0) > 0.02 for c in FEATURE_GROUPS['dns'] if 'z_score' in c or 'cv_' in c or 'trend' in c),
                'context_matters': group_totals['contextual'] > 0.05,
                'provider_change_context_matters': group_totals['contexto_cambios'] > 0.05,
                'composite_score_dominates': group_totals['score_compuesto'] > 0.30,
            },
        }

    # ════════════════════════════════════════════════════════════════════
    # Predicción — devuelve probabilidad por clase (no solo failover)
    # ════════════════════════════════════════════════════════════════════
    def predict_decision_proba(self, metrics_dict):
        """Predice P(clase) para las 4 clases de decisión, dado un dict de features."""
        if self.model is None or self.features_used is None:
            raise ValueError("Entrenar el modelo primero")

        missing = [f for f in self.features_used if f not in metrics_dict]
        if missing:
            logger.warning(f"⚠️ Faltan {len(missing)} features: {missing}")
            for f in missing:
                metrics_dict[f] = 0

        row = dict(metrics_dict)
        for col, mapping in CATEGORICAL_FEATURES.items():
            if col in row and isinstance(row[col], str):
                row[col] = mapping.get(row[col], -1)

        X = pd.DataFrame([row])[self.features_used]
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
        # Mismo fix que en prepare_features: forzar numérico, ej. si algún
        # valor llegó como None desde el caller.
        object_cols = X.select_dtypes(include='object').columns.tolist()
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        proba = self.model.predict_proba(X)[0]
        return dict(zip(self.label_encoder.classes_, proba.tolist()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("═" * 80)
    logger.info("xgboost_optimizer.py v3 — Flujo recomendado (multiclase, Opción A):")
    logger.info("═" * 80)
    logger.info("")
    logger.info("  from xgboost_optimizer import ScoringWeightOptimizer")
    logger.info("")
    logger.info("  optimizer = ScoringWeightOptimizer()")
    logger.info("  best_params = optimizer.tune_hyperparameters(df, n_trials=15)")
    logger.info("  optimizer.train_with_cv(df, best_params=best_params)")
    logger.info("  weights = optimizer.get_optimized_weights()")
    logger.info("  proba = optimizer.predict_decision_proba(metrics_dict)")
    logger.info("═" * 80)
