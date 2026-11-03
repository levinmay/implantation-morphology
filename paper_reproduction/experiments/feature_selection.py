"""Training-only feature ranking for the frozen five-fold staging evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

SEED = 20260827


def validate_frame(frame):
    if frame.unit_id.duplicated().any():
        raise ValueError('Duplicate observation')
    if set(frame.evaluation_fold) != set(range(5)):
        raise ValueError('Expected five folds')
    if frame[['unit_id', 'component_id', 'evaluation_fold', 'target']].isna().any().any():
        raise ValueError('Missing identity, fold or label')
    if (frame.groupby('component_id').evaluation_fold.nunique() != 1).any():
        raise ValueError('A linkage group crosses folds')
    if set(frame.target) != set(range(4)):
        raise ValueError('Expected four classes')


def auc(estimator, x, y):
    if list(estimator.classes_) != list(range(4)):
        raise ValueError('A fitting partition is missing a class')
    return float(roc_auc_score(y, estimator.predict_proba(x), multi_class='ovr',
                              average='macro', labels=np.arange(4)))


def rank_training_features(train, columns, outer, *, trees=500, repeats=10, jobs=4):
    folds = sorted(train.evaluation_fold.unique())
    if len(folds) != 3 or outer in folds or (outer + 1) % 5 in folds:
        raise ValueError('Ranking must use exactly the three training folds')
    if len(columns) != len(set(columns)) or not columns:
        raise ValueError('Invalid candidate columns')
    if np.isinf(train[columns].to_numpy(float)).any():
        raise ValueError('Infinite training feature')
    # Selection of eligible columns uses training data only, never test/validation.
    usable = sorted(c for c in columns if train[c].nunique(dropna=True) > 1)
    if not usable:
        raise ValueError('No nonconstant training features')
    records = []
    for inner in folds:
        fitting = train.loc[train.evaluation_fold != inner]
        held = train.loc[train.evaluation_fold == inner]
        if set(fitting.component_id) & set(held.component_id):
            raise ValueError('Ranking group overlap')
        for part in (fitting, held):
            if set(part.target) != set(range(4)):
                raise ValueError('Ranking split does not contain every class')
        seed = SEED + 100 * outer + int(inner)
        model = Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('classify', ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=5,
                 max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=jobs)),
        ])
        model.fit(fitting[usable], fitting.target)
        importance = permutation_importance(model, held[usable], held.target, scoring=auc,
                    n_repeats=repeats, random_state=seed + 10000, n_jobs=1)
        for j, feature in enumerate(usable):
            records.append({'outer_fold': outer, 'inner_fold': int(inner),
                            'feature': feature, 'mean_auc_drop': float(importance.importances_mean[j])})
    detail = pd.DataFrame(records)
    ranking = detail.groupby('feature', as_index=False).mean(numeric_only=True)
    ranking = ranking[['feature', 'mean_auc_drop']].sort_values(
        ['mean_auc_drop', 'feature'], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    return ranking, detail, sorted(set(columns) - set(usable))


def fit_outer(frame, columns, budget, outer, candidate_factory, classes, *,
              trees=500, repeats=10, jobs=4):
    validate_frame(frame)
    val_fold = (outer + 1) % 5
    train = frame.loc[~frame.evaluation_fold.isin([outer, val_fold])].copy()
    val = frame.loc[frame.evaluation_fold == val_fold].copy()
    test = frame.loc[frame.evaluation_fold == outer].copy()
    ranking, detail, excluded = rank_training_features(
        train, columns, outer, trees=trees, repeats=repeats, jobs=jobs)
    if len(ranking) < budget:
        raise ValueError('Fewer eligible training features than the fixed budget')
    features = ranking.head(budget).feature.tolist()
    trials = []
    best = None
    for name, estimator in candidate_factory(features, outer):
        estimator.fit(train[features], train.target)
        score = auc(estimator, val[features], val.target)
        trials.append({'model': name, 'validation_macro_auroc': score})
        if best is None or (score, name) > (best[0], best[1]):
            best = (score, name, estimator)
    if best is None:
        raise ValueError('No classifier candidates')
    score, name, estimator = best
    development = pd.concat([train, val], ignore_index=True)
    estimator.fit(development[features], development.target)
    if list(estimator.classes_) != list(range(4)):
        raise ValueError('Unexpected final class order')
    probability = estimator.predict_proba(test[features])
    predicted = test[['unit_id', 'component_id', 'evaluation_fold', 'target', 'four_group']].copy()
    for j, label in enumerate(classes):
        predicted['prob_' + label] = probability[:, j]
    record = {'fold': outer, 'validation_fold': val_fold, 'selected_model': name,
              'validation_macro_auroc': score, 'feature_count': budget,
              'features': features, 'excluded_constant_or_missing': excluded,
              'train_units': len(train), 'validation_units': len(val), 'test_units': len(test),
              'morphology_features': sum(f.startswith('rich_') for f in features),
              'appearance_features': sum(f.startswith('texture_') for f in features)}
    return predicted, record, ranking, detail, trials, estimator
