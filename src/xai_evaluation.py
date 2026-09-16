"""Model-behavior experiments; no causal or physical-actionability claims."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.explain_shap import _positive_class_matrix
from src.utils import PROJECT_ROOT


def artifact_fingerprint(models_dir=None) -> str:
    """Bind reports and UI caches to data, artifacts, and experiment implementation."""
    directory = Path(models_dir) if models_dir is not None else PROJECT_ROOT / 'models'
    files = sorted(directory.glob('*.pkl')) + [directory/'metadata.json'] + sorted((PROJECT_ROOT / 'src').glob('*.py'))
    files += sorted((PROJECT_ROOT / 'data').glob('*.data')) + [PROJECT_ROOT / 'run_experiments.py']
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def similarities(a, b, k=5):
    a, b = np.asarray(a), np.asarray(b)
    k = min(k, len(a))
    sa, sb = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(b))[:k])
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    corr = spearmanr(np.abs(a), np.abs(b)).statistic if np.std(np.abs(a)) and np.std(np.abs(b)) else None
    return {'cosine': float(np.dot(a, b) / norm) if norm else None,
            'rank_correlation': float(corr) if corr is not None and np.isfinite(corr) else None,
            'top_k_overlap': len(sa & sb) / k, 'jaccard': len(sa & sb) / len(sa | sb)}


def faithfulness(model, row, shap_values, training, k=5, repeats=30, seed=42):
    """Replace top-|SHAP| features with training medians; matched random-k control.

    Score = |p(original)-p(top-k replaced)| - mean |p(original)-p(random-k replaced)|.
    Positive scores favor the explanation, negative scores remain visible.
    """
    rng = np.random.default_rng(seed)
    k = min(k, row.shape[1])
    median = training.median().to_numpy()
    original = row.iloc[0].to_numpy().copy()
    top = np.argsort(-np.abs(shap_values))[:k]
    changed = original.copy(); changed[top] = median[top]
    random_rows = np.tile(original, (repeats, 1))
    for r in random_rows:
        indices = rng.choice(len(original), k, replace=False)
        r[indices] = median[indices]
    batch = pd.DataFrame(np.vstack([original, changed, random_rows]), columns=row.columns)
    probs = model.predict_proba(batch)[:, 1]
    random_effects = np.abs(probs[2:] - probs[0])
    return {'original_probability': float(probs[0]), 'explained_probability': float(probs[1]),
            'random_probability': float(probs[2:].mean()),
            'explained_change': float(abs(probs[1] - probs[0])),
            'random_change': float(random_effects.mean()),
            'random_change_std': float(random_effects.std()),
            'faithfulness': float(abs(probs[1] - probs[0]) - random_effects.mean()),
            'features': list(row.columns[top]), 'k': k, 'random_repeats': repeats}


def evaluate_explanations(pipeline, training, test, sample_count=30, seed=42):
    from src.explain_shap import SHAPExplainer
    from src.explain_lime import LIMEExplainer
    rng = np.random.default_rng(seed)
    sample = test.sample(min(sample_count, len(test)), random_state=seed)
    shap = SHAPExplainer(pipeline.model, training)
    values = _positive_class_matrix(shap.explain(sample).values)
    probs = pipeline.model.predict_proba(sample)[:, 1]
    scale = training.std().replace(0, 1).to_numpy()
    noise = rng.normal(size=sample.shape) * scale
    rows, robustness, vectors = [], [], []
    # The same noise direction at each magnitude isolates magnitude effects.
    for level in [0., .01, .05, .10]:
        perturbed = pd.DataFrame(sample.to_numpy() + level * noise, columns=sample.columns, index=sample.index)
        perturbed_values = _positive_class_matrix(shap.explain(perturbed).values)
        perturbed_probs = pipeline.model.predict_proba(perturbed)[:, 1]
        for i, idx in enumerate(sample.index):
            sim = similarities(values[i], perturbed_values[i])
            robustness.append({'sample_id': int(idx), 'perturbation': level, **sim,
                               'prediction_stable': int((probs[i] >= pipeline.default_threshold) == (perturbed_probs[i] >= pipeline.default_threshold)),
                               'probability_change': float(abs(probs[i] - perturbed_probs[i]))})
            if level == .01:
                vectors.append({'sample_id': int(idx), 'original': values[i].tolist(), 'perturbed': perturbed_values[i].tolist()})
    for i, idx in enumerate(sample.index):
        row = sample.loc[[idx]]
        faith = faithfulness(pipeline.model, row, values[i], training, seed=seed+i)
        lime = LIMEExplainer(training, random_state=seed).explain_instance(pipeline.model, row, num_features=10)
        lime_values = np.zeros(sample.shape[1])
        for c in lime['contributors']:
            lime_values[sample.columns.get_loc(c['feature'])] = c['contribution']
        consistency = similarities(values[i], lime_values)
        stable = next(r for r in robustness if r['sample_id'] == int(idx) and r['perturbation'] == .01)
        confidence = max(float(probs[i]), 1-float(probs[i]))
        rows.append({'sample_id': int(idx), **faith, 'confidence': confidence,
                     'confidence_group': 'Low [0.5,0.7)' if confidence < .7 else 'Medium [0.7,0.9)' if confidence < .9 else 'High [0.9,1]',
                     'stability': stable['cosine'], 'fidelity_r2': lime['local_model_r2'],
                     'surrogate_probability': lime['surrogate_probability'],
                     'consistency_jaccard': consistency['jaccard'], 'consistency_overlap': consistency['top_k_overlap'],
                     'consistency_rank_correlation': consistency['rank_correlation']})
    frame = pd.DataFrame(rows)
    return {'samples': rows, 'robustness': robustness, 'stability_vectors': vectors,
            'features': list(sample.columns),
            'summary': {'faithfulness': float(frame.faithfulness.mean()), 'fidelity': float(frame.fidelity_r2.mean()),
                        'stability': float(frame.stability.mean()), 'consistency': float(frame.consistency_jaccard.mean()),
                        'robustness': float(pd.DataFrame(robustness).query('perturbation == 0.10').cosine.mean())},
            'confidence_groups': frame.groupby('confidence_group').agg(count=('sample_id','size'), confidence=('confidence','mean'),
                       faithfulness=('faithfulness','mean'), stability=('stability','mean')).reset_index().to_dict('records')}
