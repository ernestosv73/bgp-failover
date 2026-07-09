#!/usr/bin/env python3
"""
logistic_regression_optimizer.py — Opción B: coeficientes lineales interpretables
═══════════════════════════════════════════════════════════════════════════════
Complementa xgboost_optimizer.py (que da RANKING de importancia vía Gain, sobre
un ensamble no-lineal) con un segundo modelo cuyos coeficientes SÍ son
directamente comparables a los pesos de la fórmula actual
(peer×0.27 + dns×0.50 + jitter×0.23).

Por qué Logistic Regression y no otra cosa (ver slide "Regresión Logística"):
├─ Misma forma matemática que la fórmula: combinación lineal de features
│   normalizadas → una decisión. Los coeficientes SON los pesos.
├─ Funciona bien con las features ya normalizadas de Etapa 1 (peer_norm,
│   dns1_norm, etc.) — no requiere ingeniería adicional.
└─ Regularización L1/L2 ayuda a evitar sobreajuste — relevante dado el tamaño
   todavía chico del dataset.

✅ REUTILIZA — no reimplementa — la preparación de features de
xgboost_optimizer.py (ScoringWeightOptimizer.prepare_features): mismo set de
~62 features, misma exclusión de fuga de datos (degradation_cycle,
provider_changed), mismo encoding de categóricas. Sin esto, comparar
coeficientes de un modelo contra la importancia del otro no sería válido —
estarían mirando espacios de features distintos.

⚠️ DIFERENCIA CLAVE respecto a XGBoost — coeficientes POR CLASE:
Con 4 clases (multinomial), Logistic Regression entrega un vector de
coeficientes DISTINTO por cada clase (normal/degradacion/failover/retorno).
No hay "un" coeficiente para dns1_norm — hay uno para su efecto sobre
P(failover), otro sobre P(retorno), etc. Esto es en realidad una ventaja
sobre el ranking único de XGBoost: permite ver si peer/dns/jitter pesan
distinto según la DIRECCIÓN de la decisión (¿el mismo balance de pesos sirve
para failover Y para retorno, o deberían ser fórmulas separadas? — ver
Opción B pendiente en la conversación).

FLUJO RECOMENDADO:
  optimizer = InterpretableWeightOptimizer()
  optimizer.train_with_cv(df, n_splits=5)
  weights_failover = optimizer.get_interpretable_weights(target_class='failover')
  weights_retorno  = optimizer.get_interpretable_weights(target_class='retorno')
  optimizer.compare_classes()   # tabla peer/dns/jitter por las 4 clases
═══════════════════════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import logging
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# sklearn >=1.8 avisa que 'penalty' se reemplazará por l1_ratio en 1.10 —
# es solo un aviso de API futura, el comportamiento actual es correcto.
warnings.filterwarnings('ignore', message=".*'penalty' was deprecated.*", category=FutureWarning)


from xgboost_optimizer import (
    ScoringWeightOptimizer, DECISION_CLASSES, FEATURE_GROUPS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpretableWeightOptimizer:
    """
    Logistic Regression multinomial + Cross-Validation sobre las mismas
    features que xgboost_optimizer.py, para extraer coeficientes
    directamente comparables a los pesos de la fórmula de scoring.
    """

    def __init__(self, penalty='l2', C=1.0):
        """
        penalty: 'l2' (default, rápido, solver lbfgs) o 'l1'/'elasticnet'
                 (selección de features vía solver saga, más lento).
        C: inverso de la fuerza de regularización (menor C = más regularización).
        """
        self.penalty = penalty
        self.C = C
        # ⚠️ sklearn >=1.5 deprecó (y >=1.7 eliminó) el parámetro multi_class:
        # con solver='lbfgs'/'saga' y más de 2 clases, ahora usa softmax
        # multinomial automáticamente — no hace falta pedirlo explícitamente.
        self.solver = 'lbfgs' if penalty == 'l2' else 'saga'

        self.model = None
        self.scaler = None
        self.features_used = None
        self.decision_classes = DECISION_CLASSES
        self.coef_by_class = None       # DataFrame: filas=features, cols=clases
        self.coef_std_by_class = None
        self.cv_scores = None
        self._feature_helper = ScoringWeightOptimizer()  # reutiliza prepare_features

    # ════════════════════════════════════════════════════════════════════
    def prepare_features(self, df):
        """Delega en ScoringWeightOptimizer.prepare_features — mismo set de
        features/exclusiones que XGBoost, para que ambos modelos sean
        directamente comparables."""
        X, y_encoded, features = self._feature_helper.prepare_features(df)
        return X, y_encoded, features

    # ════════════════════════════════════════════════════════════════════
    def train_with_cv(self, df, n_splits=5, random_state=42):
        logger.info("\n" + "═" * 80)
        logger.info(f"📐 Logistic Regression multinomial (penalty={self.penalty}, C={self.C}): Cross-Validation")
        logger.info("═" * 80)

        X, y_encoded, features = self.prepare_features(df)
        self.features_used = features

        class_counts = np.bincount(y_encoded, minlength=len(self.decision_classes))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < n_splits:
            logger.warning(f"⚠️ Clase minoritaria tiene {min_class_count} muestras "
                           f"para {n_splits} folds → reduciendo a {max(2, min_class_count)}")
            n_splits = max(2, min_class_count)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_metrics = {'precision_macro': [], 'recall_macro': [], 'f1_macro': [], 'roc_auc_ovr': []}
        fold_coefs = []   # cada elemento: array (n_classes, n_features) de ese fold

        logger.info(f"\n🔄 {n_splits}-Fold Stratified CV...")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # ⚠️ Escalar SOLO con estadísticas del fold de entrenamiento —
            # escalar con todo el dataset antes de splitear sería fuga de datos.
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression(
                solver=self.solver,
                penalty=self.penalty,
                C=self.C,
                class_weight='balanced',
                max_iter=2000,
                random_state=random_state,
            )
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)

            fold_metrics['precision_macro'].append(
                precision_score(y_test, y_pred, average='macro', zero_division=0))
            fold_metrics['recall_macro'].append(
                recall_score(y_test, y_pred, average='macro', zero_division=0))
            fold_metrics['f1_macro'].append(
                f1_score(y_test, y_pred, average='macro', zero_division=0))

            try:
                present_classes = np.unique(y_test)
                if len(present_classes) >= 2:
                    fold_metrics['roc_auc_ovr'].append(
                        roc_auc_score(y_test, y_pred_proba, multi_class='ovr',
                                       average='macro', labels=list(range(len(self.decision_classes))))
                    )
                else:
                    fold_metrics['roc_auc_ovr'].append(np.nan)
            except Exception as e:
                logger.warning(f"   ⚠️ ROC-AUC no calculable en fold {fold}: {e}")
                fold_metrics['roc_auc_ovr'].append(np.nan)

            # model.coef_ shape: (n_classes, n_features) — orden de clases =
            # model.classes_ (coincide con el orden del LabelEncoder porque
            # y_encoded ya viene codificado 0..3 en ese orden fijo)
            fold_coefs.append(model.coef_.copy())

            logger.info(
                f"   Fold {fold}/{n_splits}: "
                f"Precision={fold_metrics['precision_macro'][-1]:.3f} | "
                f"Recall={fold_metrics['recall_macro'][-1]:.3f} | "
                f"F1={fold_metrics['f1_macro'][-1]:.3f} | "
                f"ROC-AUC(OvR)={fold_metrics['roc_auc_ovr'][-1]:.3f}"
            )

        self.cv_scores = fold_metrics

        logger.info(f"\n📊 Cross-Validation Results ({n_splits} folds, macro-average):")
        logger.info("-" * 80)
        for metric, values in fold_metrics.items():
            valid = [v for v in values if not np.isnan(v)]
            if valid:
                logger.info(f"   {metric:<18}: {np.mean(valid):.4f} ± {np.std(valid):.4f}")

        # Promediar coeficientes entre folds (misma idea que feature_importance
        # promediada en XGBoost)
        stacked = np.stack(fold_coefs, axis=0)   # (n_folds, n_classes, n_features)
        mean_coefs = stacked.mean(axis=0)
        std_coefs = stacked.std(axis=0)

        self.coef_by_class = pd.DataFrame(
            mean_coefs.T, index=features, columns=self.decision_classes
        )
        self.coef_std_by_class = pd.DataFrame(
            std_coefs.T, index=features, columns=self.decision_classes
        )

        # Modelo final con todos los datos (para predict_decision_proba)
        logger.info(f"\n🔄 Entrenando modelo final (todos los datos)...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(
            solver=self.solver, penalty=self.penalty,
            C=self.C, class_weight='balanced', max_iter=2000, random_state=random_state,
        )
        self.model.fit(X_scaled, y_encoded)

        return self.coef_by_class

    # ════════════════════════════════════════════════════════════════════
    def get_interpretable_weights(self, target_class='failover'):
        """
        Traduce los coeficientes de UNA clase a pesos candidatos peer/dns/
        jitter, directamente comparables a la fórmula actual (0.27/0.50/0.23).

        Usa VALOR ABSOLUTO del coeficiente (magnitud del efecto sobre esa
        clase, sin importar el signo) agrupado por FEATURE_GROUPS — misma
        taxonomía que xgboost_optimizer.py, para comparabilidad directa.
        """
        if self.coef_by_class is None:
            logger.error("❌ Ejecutar train_with_cv() primero")
            return None
        if target_class not in self.decision_classes:
            raise ValueError(f"target_class debe ser uno de {self.decision_classes}")

        logger.info("\n" + "═" * 80)
        logger.info(f"📈 PESOS INTERPRETABLES — clase '{target_class}' (coeficientes Logistic Regression)")
        logger.info("═" * 80)

        coefs = self.coef_by_class[target_class]
        abs_coefs = coefs.abs()

        logger.info(f"\n🔍 Top 15 features (|coeficiente|, clase '{target_class}'):")
        top15 = abs_coefs.sort_values(ascending=False).head(15)
        for idx, (feat, val) in enumerate(top15.items(), 1):
            sign = "➕" if coefs[feat] > 0 else "➖"
            bar = "█" * int(val * 10 / max(top15.max(), 0.001))
            logger.info(f"   {idx:2d}. {sign} {feat:28s} {bar:12s} {val:.4f} "
                        f"(± {self.coef_std_by_class.loc[feat, target_class]:.4f})")

        group_totals = {}
        for group, cols in FEATURE_GROUPS.items():
            group_totals[group] = float(sum(abs_coefs.get(c, 0.0) for c in cols))

        logger.info(f"\n📋 RESUMEN POR FAMILIA (|coeficiente| sumado, clase '{target_class}'):")
        total_all = sum(group_totals.values()) or 1.0
        for group, total in sorted(group_totals.items(), key=lambda x: -x[1]):
            pct = total / total_all * 100
            bar = "█" * int(pct / 2)
            logger.info(f"   {group:20s} {bar:25s} {pct:5.1f}%")

        core = group_totals['peer'] + group_totals['dns'] + group_totals['jitter']
        candidate_peer_w = group_totals['peer'] / core if core > 0 else 0.27
        candidate_dns_w = group_totals['dns'] / core if core > 0 else 0.50
        candidate_jitter_w = group_totals['jitter'] / core if core > 0 else 0.23

        logger.info(f"\n🎯 Pesos candidatos (renormalizados peer+dns+jitter=1, clase '{target_class}'):")
        logger.info(f"   peer   = {candidate_peer_w:.4f}  (actual: 0.27)")
        logger.info(f"   dns    = {candidate_dns_w:.4f}  (actual: 0.50)")
        logger.info(f"   jitter = {candidate_jitter_w:.4f}  (actual: 0.23)")

        return {
            'target_class': target_class,
            'group_totals': group_totals,
            'candidate_weights': {
                'peer': float(candidate_peer_w),
                'dns': float(candidate_dns_w),
                'jitter': float(candidate_jitter_w),
            },
            'current_weights': {'peer': 0.27, 'dns': 0.50, 'jitter': 0.23},
            'top_features': top15.to_dict(),
            'coefficients_signed': coefs.to_dict(),
        }

    # ════════════════════════════════════════════════════════════════════
    def compare_classes(self):
        """
        Tabla peer/dns/jitter (pesos candidatos renormalizados) para las 4
        clases lado a lado. Si failover y retorno dan balances muy distintos,
        es evidencia de que una única fórmula compartida podría no ser
        óptima para ambas direcciones (ver Opción B pendiente).
        """
        if self.coef_by_class is None:
            logger.error("❌ Ejecutar train_with_cv() primero")
            return None

        logger.info("\n" + "═" * 80)
        logger.info("📊 COMPARACIÓN peer/dns/jitter POR CLASE (¿misma fórmula sirve para todas?)")
        logger.info("═" * 80)

        rows = []
        for cls in self.decision_classes:
            coefs = self.coef_by_class[cls].abs()
            group_totals = {g: float(sum(coefs.get(c, 0.0) for c in cols))
                             for g, cols in FEATURE_GROUPS.items()}
            core = group_totals['peer'] + group_totals['dns'] + group_totals['jitter']
            rows.append({
                'clase': cls,
                'peer': group_totals['peer'] / core if core > 0 else np.nan,
                'dns': group_totals['dns'] / core if core > 0 else np.nan,
                'jitter': group_totals['jitter'] / core if core > 0 else np.nan,
            })

        comparison_df = pd.DataFrame(rows).set_index('clase')
        logger.info(f"\n{comparison_df.round(4).to_string()}")

        spread = comparison_df[['peer', 'dns', 'jitter']].std(axis=0)
        logger.info(f"\n   Dispersión entre clases (std): peer={spread['peer']:.3f} "
                    f"dns={spread['dns']:.3f} jitter={spread['jitter']:.3f}")
        if spread.max() > 0.15:
            logger.info("   ⚠️ Dispersión notable entre clases — el balance peer/dns/jitter "
                        "cambia según la decisión. Podría justificar separar la fórmula "
                        "por dirección (failover vs. retorno) en una futura iteración.")
        else:
            logger.info("   ✅ Balance relativamente consistente entre clases — apoya "
                        "mantener una única fórmula compartida.")

        return comparison_df

    # ════════════════════════════════════════════════════════════════════
    def predict_decision_proba(self, metrics_dict):
        if self.model is None or self.scaler is None:
            raise ValueError("Entrenar el modelo primero")

        missing = [f for f in self.features_used if f not in metrics_dict]
        for f in missing:
            metrics_dict[f] = 0

        from xgboost_optimizer import CATEGORICAL_FEATURES
        row = dict(metrics_dict)
        for col, mapping in CATEGORICAL_FEATURES.items():
            if col in row and isinstance(row[col], str):
                row[col] = mapping.get(row[col], -1)

        X = pd.DataFrame([row])[self.features_used]
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
        # Mismo fix que en prepare_features: primero fillna(0) general (cubre
        # NaN genuino en columnas ya float64), LUEGO forzar numérico en
        # cualquier columna que haya quedado en dtype object.
        X = X.fillna(0)
        object_cols = X.select_dtypes(include='object').columns.tolist()
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        return dict(zip(self.decision_classes, proba.tolist()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("═" * 80)
    logger.info("logistic_regression_optimizer.py — Flujo recomendado (Opción B):")
    logger.info("═" * 80)
    logger.info("")
    logger.info("  from logistic_regression_optimizer import InterpretableWeightOptimizer")
    logger.info("")
    logger.info("  optimizer = InterpretableWeightOptimizer(penalty='l2', C=1.0)")
    logger.info("  optimizer.train_with_cv(df, n_splits=5)")
    logger.info("  weights = optimizer.get_interpretable_weights(target_class='failover')")
    logger.info("  optimizer.compare_classes()")
    logger.info("═" * 80)
