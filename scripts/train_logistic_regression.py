#!/usr/bin/env python3
"""
train_logistic_regression.py — Opción B completa: XGBoost + Logistic Regression
✅ Corre AMBOS modelos sobre el mismo dataset (ml_features) y produce la tabla
   comparativa final: fórmula actual vs. candidato XGBoost (Gain) vs.
   candidato Logistic Regression (coeficientes).

Reutiliza load_training_data_from_ml_features() de train_from_ml_features.py
— misma query, mismo dataset, para que la comparación entre los tres sea
sobre exactamente los mismos datos.
"""
import logging
import numpy as np

from train_from_ml_features import load_training_data_from_ml_features
from xgboost_optimizer import ScoringWeightOptimizer, DECISION_CLASSES
from logistic_regression_optimizer import InterpretableWeightOptimizer

logger = logging.getLogger(__name__)


def print_comparison_table(current, xgb_weights, lr_weights_failover, lr_weights_retorno):
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN FINAL — fórmula actual vs. candidatos (Opción B)")
    print("=" * 80)
    print(f"\n{'':12s}{'peer':>10s}{'dns':>10s}{'jitter':>10s}")
    print(f"{'Actual':12s}{current['peer']:>10.4f}{current['dns']:>10.4f}{current['jitter']:>10.4f}")
    print(f"{'XGBoost':12s}{xgb_weights['peer']:>10.4f}{xgb_weights['dns']:>10.4f}{xgb_weights['jitter']:>10.4f}")
    print(f"{'LR(failover)':12s}{lr_weights_failover['peer']:>10.4f}{lr_weights_failover['dns']:>10.4f}{lr_weights_failover['jitter']:>10.4f}")
    print(f"{'LR(retorno)':12s}{lr_weights_retorno['peer']:>10.4f}{lr_weights_retorno['dns']:>10.4f}{lr_weights_retorno['jitter']:>10.4f}")

    print("\n⚠️ Lectura correcta de esta tabla:")
    print("   - XGBoost: ranking de importancia (Gain) — orden, no magnitud lineal.")
    print("   - LR: coeficientes reales de una combinación lineal — SÍ comparables")
    print("     en forma matemática a los pesos de la fórmula actual.")
    print("   - Si XGBoost y LR apuntan en la misma dirección (ambos suben 'dns',")
    print("     por ejemplo), es una señal más confiable que si solo uno lo sugiere.")
    print("   - Estos números son de UN dataset todavía chico/mayormente sintético")
    print("     — tratarlos como hipótesis a validar con más datos, no como el")
    print("     reemplazo definitivo de 0.27/0.50/0.23.")


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Opción B completa: XGBoost + Logistic Regression desde ml_features')
    parser.add_argument('--days', type=int, default=None,
                         help='Límite de días hacia atrás (default: None = toda la tabla, sin límite)')
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 OPCIÓN B COMPLETA — XGBoost (ranking) + Logistic Regression (coeficientes)")
    print("=" * 80)

    print("\nPASO 1: Cargar datos de ml_features")
    print("-" * 80)
    df = load_training_data_from_ml_features(timescaledb_password='bgp_app_password', days=args.days)
    if df.empty:
        print("❌ Sin datos — abortando.")
        return
    print(f"\n✅ Dataset cargado: {len(df)} registros")

    # ── XGBoost ──────────────────────────────────────────────────────────
    print("\nPASO 2: Entrenar XGBoost (ranking de importancia)")
    print("-" * 80)
    xgb_optimizer = ScoringWeightOptimizer()
    best_params = xgb_optimizer.tune_hyperparameters(df, n_trials=15)
    xgb_optimizer.train_with_cv(df, n_splits=5, best_params=best_params)
    xgb_result = xgb_optimizer.get_optimized_weights()

    # ── Logistic Regression ─────────────────────────────────────────────
    print("\nPASO 3: Entrenar Logistic Regression (coeficientes interpretables)")
    print("-" * 80)
    lr_optimizer = InterpretableWeightOptimizer(penalty='l2', C=1.0)
    lr_optimizer.train_with_cv(df, n_splits=5)
    lr_failover = lr_optimizer.get_interpretable_weights(target_class='failover')
    lr_retorno = lr_optimizer.get_interpretable_weights(target_class='retorno')
    lr_optimizer.compare_classes()

    # ── Comparación final ────────────────────────────────────────────────
    print_comparison_table(
        current={'peer': 0.27, 'dns': 0.50, 'jitter': 0.23},
        xgb_weights=xgb_result['candidate_weights'],
        lr_weights_failover=lr_failover['candidate_weights'],
        lr_weights_retorno=lr_retorno['candidate_weights'],
    )

    print("\nMétricas de ambos modelos (macro-average, para contexto de qué tan")
    print("confiables son estos números):")
    print(f"\n{'':20s}{'Precision':>12s}{'Recall':>12s}{'F1':>12s}{'ROC-AUC':>12s}")
    for name, scores in [('XGBoost', xgb_result['cv_scores']), ('Logistic Reg.', lr_optimizer.cv_scores)]:
        vals = {k: np.mean([v for v in vv if not np.isnan(v)]) for k, vv in scores.items()}
        print(f"{name:20s}{vals['precision_macro']:>12.4f}{vals['recall_macro']:>12.4f}"
              f"{vals['f1_macro']:>12.4f}{vals['roc_auc_ovr']:>12.4f}")

    print("\n" + "=" * 80)
    print("✅ OPCIÓN B COMPLETADA")
    print("=" * 80)


if __name__ == '__main__':
    main()
