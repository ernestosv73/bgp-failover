#!/usr/bin/env python3
"""
train_temporal_cv.py — Validación de 3 niveles respetando orden cronológico
═══════════════════════════════════════════════════════════════════════════════
Variante NUEVA, en paralelo a xgboost_optimizer.py / logistic_regression_optimizer.py
(que usan StratifiedKFold) — NO los reemplaza. Pensado para correr ambos
esquemas y comparar resultados directamente (ver conversación).

✅ POR QUÉ EXISTE: StratifiedKFold mezcla filas de cualquier semana del
dataset sin respetar el orden cronológico — con series temporales, eso
puede significar entrenar con datos de una semana posterior a la que se
usa para validar (filtración de "futuro" hacia "pasado"). Este script
resuelve eso con un esquema de 3 niveles, donde cada nivel ataca un solo
problema (a diferencia de Stratified, que le pedía a un solo mecanismo
resolver dos problemas en tensión: orden cronológico Y balance de clases):

  NIVEL 1 — Holdout temporal 80/20:
      El 20% cronológicamente más reciente se aparta y NUNCA se toca
      durante el CV — solo se usa una vez, al final, para la evaluación
      de generalización genuina.

  NIVEL 2 — Walk-Forward (TimeSeriesSplit, ventana expansiva) DENTRO del 80%:
      Cada fold entrena con TODO el pasado disponible hasta ese punto y
      valida con el tramo cronológico inmediatamente siguiente — nunca al
      revés.

  NIVEL 3 — class weights (compute_sample_weight('balanced', ...)):
      Mismo mecanismo YA usado en el pipeline original — resuelve el
      desbalance de clases SIN necesitar que el split rompa el orden
      cronológico para lograrlo (a diferencia de Stratified, que logra el
      balance decidiendo QUÉ FILAS van a cada fold).

⚠️ LIMITACIÓN CONOCIDA (validada empíricamente antes de implementar esto,
ver conversación): con clases tan escasas como failover/retorno, los folds
más tempranos (menos historia acumulada) pueden quedar con muy pocas
muestras — no cero, pero sí una estimación más ruidosa para esos folds
puntuales. Los class weights compensan escasez relativa, no ausencia total.

Requiere los mismos datos que el pipeline original (ml_features) — no
necesita ninguna tabla ni migración nueva.

USO:
    python3 train_temporal_cv.py
    python3 train_temporal_cv.py --model both --n-splits 5 --holdout-fraction 0.2
    python3 train_temporal_cv.py --days 60
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from xgboost_optimizer import ScoringWeightOptimizer, DECISION_CLASSES, FEATURE_GROUPS
from train_from_ml_features import load_training_data_from_ml_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def temporal_holdout_split(df, holdout_fraction=0.2):
    """
    NIVEL 1 — Corte cronológico simple: el df YA viene ordenado por 'time'
    (ver load_training_data_from_ml_features, ORDER BY time) — no hace
    falta reordenar, solo cortar por posición de índice.
    """
    cut = int(len(df) * (1 - holdout_fraction))
    df_trainval = df.iloc[:cut].reset_index(drop=True)
    df_holdout = df.iloc[cut:].reset_index(drop=True)
    logger.info(f"📐 Nivel 1 — Holdout temporal: {len(df_trainval)} filas train/val "
                f"({df_trainval['time'].min()} → {df_trainval['time'].max()}) | "
                f"{len(df_holdout)} filas holdout final "
                f"({df_holdout['time'].min()} → {df_holdout['time'].max()})")
    return df_trainval, df_holdout


def _fit_and_eval(model_kind, X_train, y_train, X_test, y_test, sample_weight, num_class, random_state=42):
    """Entrena UN modelo (xgboost o logreg) con los pesos de clase (Nivel 3) y evalúa."""
    if model_kind == 'xgboost':
        model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=num_class,
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state, verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    elif model_kind == 'logreg':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=random_state)
        model.fit(X_train_s, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)
    else:
        raise ValueError(f"model_kind desconocido: {model_kind}")

    metrics = {
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    present_classes = np.unique(y_test)
    try:
        if len(present_classes) >= 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr',
                                                average='macro', labels=list(range(num_class)))
        else:
            metrics['roc_auc'] = np.nan
    except Exception:
        metrics['roc_auc'] = np.nan

    return model, metrics


def extract_final_model_weights(model, model_kind, feature_names, target_class='failover'):
    """
    Extrae pesos candidatos peer/dns/jitter SOLO del modelo final (Nivel 1,
    entrenado con el 80% train/val completo, el que se evalúa contra el
    holdout) — misma lógica de agrupar→seleccionar→renormalizar que
    get_optimized_weights() (xgboost_optimizer.py) y get_interpretable_
    weights() (logistic_regression_optimizer.py), reutilizando FEATURE_GROUPS.

    ⚠️ Deliberadamente NO promedia entre los folds de Walk-Forward (Nivel 2):
    a diferencia de Stratified (folds disjuntos e independientes), los folds
    de Walk-Forward se SUPERPONEN — el train del fold 5 contiene TODO el
    train del fold 1 más datos nuevos, no son 5 muestras independientes.
    Promediar importancia/coeficientes entre ellos sería menos riguroso de
    lo que parece. El modelo final es además el que "quedaría desplegado"
    en la práctica — un solo ajuste, sin ambigüedad de cuál usar.
    """
    if model_kind == 'xgboost':
        imp_map = dict(zip(feature_names, model.feature_importances_))
        label = "Gain (XGBoost, SOLO modelo final — no promediado entre folds)"
    elif model_kind == 'logreg':
        # model.coef_ shape: (n_clases, n_features) — |coeficiente| de UNA
        # clase (target_class), mismo criterio que logistic_regression_optimizer.py
        class_idx = list(DECISION_CLASSES).index(target_class)
        coefs = model.coef_[class_idx]
        imp_map = dict(zip(feature_names, np.abs(coefs)))
        label = f"|coeficiente| (Logistic Regression, clase '{target_class}', SOLO modelo final)"
    else:
        raise ValueError(f"model_kind desconocido: {model_kind}")

    group_totals = {}
    for group, cols in FEATURE_GROUPS.items():
        group_totals[group] = float(sum(imp_map.get(c, 0.0) for c in cols))

    core = group_totals['peer'] + group_totals['dns'] + group_totals['jitter']
    candidate_peer_w = group_totals['peer'] / core if core > 0 else 0.27
    candidate_dns_w = group_totals['dns'] / core if core > 0 else 0.50
    candidate_jitter_w = group_totals['jitter'] / core if core > 0 else 0.23

    return {
        'label': label,
        'group_totals': group_totals,
        'peer': candidate_peer_w,
        'dns': candidate_dns_w,
        'jitter': candidate_jitter_w,
    }


def train_temporal_cv(df, model_kind='xgboost', n_splits=5, holdout_fraction=0.2, random_state=42):
    """
    Orquesta los 3 niveles completos para UN modelo. Devuelve las métricas
    por fold (Nivel 2, dentro del 80%) y la métrica final sobre el holdout
    (Nivel 1, el 20% nunca tocado durante el CV).
    """
    optimizer = ScoringWeightOptimizer()

    df_trainval, df_holdout = temporal_holdout_split(df, holdout_fraction)

    # ════════════════════════════════════════════════════════════════════
    # NIVEL 2 — Walk-Forward (TimeSeriesSplit) DENTRO del 80%
    # ════════════════════════════════════════════════════════════════════
    logger.info(f"\n📐 Nivel 2 — Walk-Forward ({n_splits} folds, ventana expansiva) dentro del 80%")
    X_trainval, y_trainval, feature_names = optimizer.prepare_features(df_trainval)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_trainval), 1):
        X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        # NIVEL 3 — class weights, calculados SOLO sobre el train de este fold
        sample_weight = compute_sample_weight('balanced', y_train)

        train_class_counts = {DECISION_CLASSES[c]: int((y_train == c).sum()) for c in range(len(DECISION_CLASSES))}
        val_class_counts = {DECISION_CLASSES[c]: int((y_val == c).sum()) for c in range(len(DECISION_CLASSES))}

        _, metrics = _fit_and_eval(model_kind, X_train, y_train, X_val, y_val,
                                    sample_weight, len(DECISION_CLASSES), random_state)
        fold_metrics.append(metrics)

        logger.info(f"   Fold {fold}/{n_splits}: train={len(train_idx):6d} (min. clase: "
                    f"{min(train_class_counts.values())}) | val={len(val_idx):6d} (min. clase: "
                    f"{min(val_class_counts.values())}) | "
                    f"Precision={metrics['precision_macro']:.3f} | Recall={metrics['recall_macro']:.3f} | "
                    f"F1={metrics['f1_macro']:.3f} | ROC-AUC={metrics['roc_auc']:.3f}")

    # ════════════════════════════════════════════════════════════════════
    # NIVEL 1 (cierre) — evaluación final sobre el 20% holdout, nunca visto
    # ════════════════════════════════════════════════════════════════════
    logger.info(f"\n📐 Nivel 1 (cierre) — Entrenar con el 80% completo, evaluar sobre el 20% holdout")
    X_holdout, y_holdout, _ = optimizer.prepare_features(df_holdout)
    # Alinear columnas por si el holdout tiene alguna categoría no vista en trainval
    X_holdout = X_holdout.reindex(columns=X_trainval.columns, fill_value=0)

    sample_weight_full = compute_sample_weight('balanced', y_trainval)
    final_model, holdout_metrics = _fit_and_eval(model_kind, X_trainval, y_trainval, X_holdout, y_holdout,
                                                  sample_weight_full, len(DECISION_CLASSES), random_state)

    logger.info(f"   HOLDOUT FINAL: Precision={holdout_metrics['precision_macro']:.3f} | "
                f"Recall={holdout_metrics['recall_macro']:.3f} | F1={holdout_metrics['f1_macro']:.3f} | "
                f"ROC-AUC={holdout_metrics['roc_auc']:.3f}")

    weights = extract_final_model_weights(final_model, model_kind, feature_names)
    logger.info(f"\n📈 Pesos candidatos — {weights['label']}")
    logger.info(f"   peer   = {weights['peer']:.4f}  (actual: 0.27)")
    logger.info(f"   dns    = {weights['dns']:.4f}  (actual: 0.50)")
    logger.info(f"   jitter = {weights['jitter']:.4f}  (actual: 0.23)")

    return fold_metrics, holdout_metrics, weights, feature_names


def main():
    parser = argparse.ArgumentParser(description='Validación de 3 niveles (temporal) — variante en paralelo a Stratified K-Fold')
    parser.add_argument('--model', choices=['xgboost', 'logreg', 'both'], default='both')
    parser.add_argument('--n-splits', type=int, default=5, help='Folds de Walk-Forward dentro del 80%% (default: 5)')
    parser.add_argument('--holdout-fraction', type=float, default=0.2,
                         help='Fracción cronológicamente más reciente apartada como test final (default: 0.2)')
    parser.add_argument('--days', type=int, default=None, help='Límite de días hacia atrás (default: toda la tabla)')
    parser.add_argument('--db-password', default='bgp_app_password')
    args = parser.parse_args()

    print("=" * 80)
    print("📐 VALIDACIÓN DE 3 NIVELES — Holdout temporal + Walk-Forward + Class weights")
    print("=" * 80)
    print("⚠️ Variante NUEVA, en paralelo al esquema Stratified K-Fold existente —")
    print("   comparar resultados entre ambos, no reemplaza al pipeline original.\n")

    print("PASO 1: Cargar datos de ml_features (ya ordenados por time)")
    print("-" * 80)
    df = load_training_data_from_ml_features(timescaledb_password=args.db_password, days=args.days)
    if df.empty:
        print("❌ Sin datos — abortando.")
        return
    print(f"✅ Dataset cargado: {len(df)} registros ({df['time'].min()} → {df['time'].max()})")

    models_to_run = ['xgboost', 'logreg'] if args.model == 'both' else [args.model]
    results = {}

    for model_kind in models_to_run:
        print("\n" + "=" * 80)
        print(f"🤖 Modelo: {model_kind.upper()}")
        print("=" * 80)
        fold_metrics, holdout_metrics, weights, feature_names = train_temporal_cv(
            df, model_kind=model_kind, n_splits=args.n_splits, holdout_fraction=args.holdout_fraction
        )
        results[model_kind] = {'folds': fold_metrics, 'holdout': holdout_metrics, 'weights': weights}

    print("\n" + "=" * 80)
    print("✅ RESUMEN — Walk-Forward (Nivel 2, dentro del 80%) vs. Holdout final (Nivel 1)")
    print("=" * 80)
    for model_kind, r in results.items():
        fold_df = pd.DataFrame(r['folds'])
        print(f"\n{model_kind.upper()}:")
        print(f"  Walk-Forward (promedio {len(r['folds'])} folds):")
        for col in ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']:
            print(f"    {col:18s}: {fold_df[col].mean():.4f} ± {fold_df[col].std():.4f}")
        print(f"  Holdout final (20%, nunca visto durante el CV):")
        for col in ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']:
            print(f"    {col:18s}: {r['holdout'][col]:.4f}")
        w = r['weights']
        print(f"  Pesos candidatos ({w['label']}):")
        print(f"    peer   = {w['peer']:.4f}  (actual: 0.27)")
        print(f"    dns    = {w['dns']:.4f}  (actual: 0.50)")
        print(f"    jitter = {w['jitter']:.4f}  (actual: 0.23)")

    print("\n⚠️ Lectura: comparar estos números contra train_from_ml_features.py/")
    print("   train_logistic_regression.py (Stratified K-Fold) para el mismo dataset —")
    print("   una brecha grande entre ambos esquemas sería evidencia de que Stratified")
    print("   estaba dando una estimación optimista por filtración temporal.")
    print("=" * 80)


if __name__ == '__main__':
    main()
