#!/usr/bin/env python3
"""Evaluate the existing trained system; reuse its preprocessing/model components."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from src.data_loader import load_secom_dataset, analyze_dataset
from src.evaluate import compute_metrics, find_best_threshold, find_threshold_for_recall, full_evaluation_report
from src.explain_shap import SHAPExplainer, _positive_class_matrix
from src.explain_dice import DiCEExplainer
from src.prediction import PredictionPipeline
from src.preprocessing import build_preprocessing_pipeline
from src.train import compare_models, get_candidate_models, build_preprocessing_and_selection_pipeline
from src.utils import MODELS_DIR, METRICS_DIR, FIGURES_DIR, EXPLANATIONS_DIR, save_json
from src.xai_evaluation import artifact_fingerprint, evaluate_explanations, similarities


def sampler(name):
    if name == 'none': return 'passthrough'
    return SMOTEENN(random_state=42) if name == 'smoteenn' else SMOTE(random_state=42)


def repair_threshold(pipeline, raw_train, y_train, raw_test, y_test):
    """Correct legacy test-tuned threshold without changing the fitted classifier."""
    full = Pipeline([(name, clone(step)) for name, step in pipeline.preprocessing_pipeline.steps] +
                    [('sampler', sampler(pipeline.metadata['imbalance_strategy'])), ('classifier', clone(pipeline.model))])
    oof = cross_val_predict(full, raw_train, y_train,
                           cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba', n_jobs=1)[:, 1]
    threshold = find_best_threshold(y_train.to_numpy(), oof)
    pipeline.default_threshold = threshold
    report = full_evaluation_report(pipeline.metadata['model_name'], y_test.to_numpy(),
                                    pipeline.predict_proba(raw_test), FIGURES_DIR, threshold)
    pipeline.metadata.update(chosen_threshold=threshold, best_f1_threshold=threshold,
        high_recall_threshold=find_threshold_for_recall(y_train.to_numpy(), oof),
        threshold_source='training-only out-of-fold predictions (5 folds)',
        test_metrics=report['metrics'], confusion_matrix=report['confusion_matrix'])
    save_json(pipeline.metadata, MODELS_DIR / 'metadata.json')
    save_json(report, METRICS_DIR / 'final_evaluation.json')
    pd.DataFrame({'sample_id': raw_train.index, 'target': y_train.to_numpy(), 'oof_probability': oof}).to_csv(METRICS_DIR / 'threshold_oof.csv', index=False)


def feature_experiments(pipeline, raw_train, y_train, raw_test, y_test):
    """Training-only SHAP selection; fixed K and threshold, no test-driven choices."""
    preprocessing = build_preprocessing_pipeline()
    train = preprocessing.fit_transform(raw_train)
    test = preprocessing.transform(raw_test)
    strategy = sampler(pipeline.metadata['imbalance_strategy'])
    balanced_x, balanced_y = (train, y_train) if strategy == 'passthrough' else strategy.fit_resample(train, y_train)
    base = clone(pipeline.model).fit(balanced_x, balanced_y)
    explainer = SHAPExplainer(base, train)
    ranking = explainer.global_importance(train.sample(min(150,len(train)), random_state=42))
    top = ranking.feature.head(40).tolist()
    removed = ranking.feature.head(5).tolist()
    configurations = [('All cleaned features', list(train.columns)), ('SHAP Top-40', top),
                      ('Remove SHAP Top-5', [f for f in train.columns if f not in removed])]
    results = []
    for name, columns in configurations:
        if name == 'All cleaned features':
            model = base
        else:
            x, y = (train[columns], y_train) if strategy == 'passthrough' else sampler(pipeline.metadata['imbalance_strategy']).fit_resample(train[columns], y_train)
            model = clone(pipeline.model).fit(x, y)
        p = model.predict_proba(test[columns])[:,1]
        results.append({'configuration': name, 'features': len(columns), 'threshold': .5,
                        **compute_metrics(y_test, (p >= .5).astype(int), p)})
    return {'results': results, 'selected_features': top, 'ablated_features': removed,
            'ranking': ranking.to_dict('records'), 'training_ids': list(map(int,raw_train.index)),
            'test_ids': list(map(int,raw_test.index)),
            'protocol': 'Same classifier hyperparameters and sampler; preprocessing and SHAP ranking fit on training only. K=40 and ablation K=5 fixed in advance. Threshold=0.5 for all configurations. All cleaned features excludes only missingness/constant columns, not MI/correlation selection. Single held-out split; exploratory, no significance or improvement claim.'}


def distribution_shift(pipeline, train, test, y_test):
    explainer = SHAPExplainer(pipeline.model, train)
    important = explainer.global_importance(train.sample(150, random_state=42)).feature.head(5).tolist()
    shifted = test.copy()
    shifted[important] += .5 * train[important].std()
    rows=[]
    original_values = _positive_class_matrix(explainer.explain(test).values)
    shifted_values = _positive_class_matrix(explainer.explain(shifted).values)
    for name, x in [('Original',test),('Shifted +0.5 training SD',shifted)]:
        p = pipeline.model.predict_proba(x)[:,1]
        rows.append({'distribution':name, **compute_metrics(y_test, (p>=pipeline.default_threshold).astype(int), p),
                     'mean_confidence':float(np.maximum(p,1-p).mean())})
    return {'results':rows, 'features_shifted':important,
            'explanation_similarity':float(np.mean([similarities(a,b)['cosine'] for a,b in zip(original_values,shifted_values)])),
            'importance':pd.DataFrame({'feature':test.columns, 'original':np.abs(original_values).mean(0),
                                       'shifted':np.abs(shifted_values).mean(0)}).to_dict('records'),
            'protocol':'Synthetic +0.5 training SD offset on five training-SHAP-ranked features; labels held fixed as a stress-test assumption. Not observed production drift or measured causal outcomes.'}


def main():
    global MODELS_DIR, METRICS_DIR, FIGURES_DIR, EXPLANATIONS_DIR
    parser=argparse.ArgumentParser(); parser.add_argument('--samples',type=int,default=30)
    parser.add_argument('--refresh-benchmark',action='store_true')
    parser.add_argument('--run',default='original',help='Saved run ID from Model Runs')
    args=parser.parse_args()
    if args.run != 'original':
        from src.model_runs import run_directory
        MODELS_DIR=run_directory(args.run)
        METRICS_DIR=MODELS_DIR/'outputs'/'metrics'
        FIGURES_DIR=MODELS_DIR/'outputs'/'figures'
        EXPLANATIONS_DIR=MODELS_DIR/'outputs'/'explanations'
        for directory in [METRICS_DIR,FIGURES_DIR,EXPLANATIONS_DIR]: directory.mkdir(parents=True,exist_ok=True)
        if args.refresh_benchmark: parser.error('--refresh-benchmark applies only to the original benchmark')
    if args.samples < 2: parser.error('--samples must be at least 2')
    pipeline=PredictionPipeline.load(MODELS_DIR/'preprocessing_pipeline.pkl',MODELS_DIR/'model.pkl',MODELS_DIR/'metadata.json')
    df=load_secom_dataset(); save_json(analyze_dataset(df).to_dict(),METRICS_DIR/'dataset_diagnostics.json')
    train=joblib.load(MODELS_DIR/'X_train_transformed.pkl'); y_train=joblib.load(MODELS_DIR/'y_train.pkl')
    raw_test=joblib.load(MODELS_DIR/'X_test_raw.pkl'); y_test=joblib.load(MODELS_DIR/'y_test.pkl')
    raw_train=df.loc[train.index,pipeline.raw_feature_names]
    assert set(raw_train.index).isdisjoint(raw_test.index)
    np.testing.assert_allclose(pipeline.transform(raw_train),train)
    if pipeline.metadata.get('threshold_source') != 'training-only out-of-fold predictions (5 folds)':
        print('Correcting threshold with training-only CV...',flush=True)
        repair_threshold(pipeline,raw_train,y_train,raw_test,y_test)
    if args.refresh_benchmark:
        print('Executing existing six-model benchmark...',flush=True)
        comparison=compare_models(raw_train,y_train,build_preprocessing_and_selection_pipeline(),get_candidate_models(),
                                  StratifiedKFold(5,shuffle=True,random_state=42),sampler_name=pipeline.metadata['imbalance_strategy'])
        comparison.to_csv(METRICS_DIR/'model_comparison_baseline.csv',index=False)
        from src.utils import load_json
        saved=load_json(MODELS_DIR/'model_comparison.json'); saved['baseline_comparison']=comparison.to_dict('records')
        save_json(saved,MODELS_DIR/'model_comparison.json')
    test=pipeline.transform(raw_test)
    print('Evaluating explanations...',flush=True)
    result=evaluate_explanations(pipeline,train,test,args.samples)
    print('Retraining explanation-guided configurations...',flush=True)
    result['feature_analysis']=feature_experiments(pipeline,raw_train,y_train,raw_test,y_test)
    result['distribution_shift']=distribution_shift(pipeline,train,test,y_test)
    shap=SHAPExplainer(pipeline.model,train)
    shap.plot_summary(train.sample(150,random_state=42),EXPLANATIONS_DIR/'shap_summary.png')
    shap.plot_bar(train.sample(150,random_state=42),EXPLANATIONS_DIR/'shap_global_importance.png')
    shap.global_importance(train.sample(150,random_state=42)).to_csv(METRICS_DIR/'shap_global_importance.csv',index=False)
    probabilities=pipeline.model.predict_proba(test)[:,1]
    demo_id=int(test.index[np.argmax(probabilities)])
    result['demo_sample_id']=demo_id
    cf=DiCEExplainer(train,y_train,pipeline.model).generate_counterfactuals(test.loc[[demo_id]],total_cfs=2,
           features_to_vary=shap.global_importance(train.sample(150,random_state=42)).feature.head(15).tolist(),
           threshold=pipeline.default_threshold)
    result['counterfactual_demo']=cf
    from src.explain_lime import LIMEExplainer
    lime = LIMEExplainer(train)
    lime_result = lime.explain_instance(pipeline.model, test.loc[[demo_id]])
    lime.plot_explanation(lime_result, EXPLANATIONS_DIR/'lime_example_explanation.png')
    (EXPLANATIONS_DIR/'lime_example_explanation.txt').write_text(lime.explain_prediction_text(pipeline.model, test.loc[[demo_id]]))
    (EXPLANATIONS_DIR/'shap_example_explanation.txt').write_text(shap.explain_prediction_text(test.loc[[demo_id]]))
    (EXPLANATIONS_DIR/'dice_example_explanation.txt').write_text(DiCEExplainer(train,y_train,pipeline.model).explain_counterfactual_text(cf))
    save_json(cf,EXPLANATIONS_DIR/'dice_example_explanation.json')
    result['created_at']=datetime.now(timezone.utc).isoformat()
    result['seed']=42; result['sample_count']=len(result['samples']); result['threshold']=pipeline.default_threshold
    result['versions']={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','shap','lime','dice-ml','xgboost','streamlit']}
    result['fingerprint']=artifact_fingerprint(MODELS_DIR)
    save_json(result,METRICS_DIR/'xai_evaluation.json')
    print('Saved real experimental results:',result['summary'],flush=True)

if __name__=='__main__': main()
