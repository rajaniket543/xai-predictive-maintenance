"""Immutable completed training runs, reusing the existing training pipeline."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import time
import uuid
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from src.utils import MODELS_DIR, PROJECT_ROOT, load_json, save_json, DEFAULT_RISK_THRESHOLDS
from src.prediction import PredictionPipeline

RUNS_DIR = MODELS_DIR / 'runs'
ARTIFACTS = ['model.pkl', 'preprocessing_pipeline.pkl', 'metadata.json', 'X_train_transformed.pkl',
             'y_train.pkl', 'X_test_raw.pkl', 'y_test.pkl']
MODEL_NAMES = {'xgboost': 'XGBoost', 'random_forest': 'Random Forest',
               'logistic_regression': 'Logistic Regression', 'hist_gradient_boosting': 'HistGradientBoosting',
               'decision_tree': 'Decision Tree'}


def run_directory(run_id='original'):
    if run_id == 'original': return MODELS_DIR
    if not run_id or Path(run_id).name != run_id or run_id.startswith('.'):
        raise ValueError('Invalid run ID')
    path = RUNS_DIR / run_id
    if not (path / 'run.json').exists(): raise ValueError('Saved run does not exist or is incomplete')
    return path


def dataset_signature():
    digest=hashlib.sha256()
    for name in ['secom.data','secom_labels.data']:
        digest.update((PROJECT_ROOT/'data'/name).read_bytes())
    return digest.hexdigest()


def split_signature(train_ids, test_ids):
    return hashlib.sha256(json.dumps([list(map(int,train_ids)),list(map(int,test_ids))]).encode()).hexdigest()


def list_runs():
    rows=[]
    for path in sorted(RUNS_DIR.glob('*/run.json'), reverse=True):
        try:
            row=load_json(path)
            if all((path.parent/name).exists() for name in ARTIFACTS): rows.append(row)
        except (OSError,ValueError): continue
    return rows


def load_run(run_id):
    directory=run_directory(run_id)
    return PredictionPipeline.load(directory/'preprocessing_pipeline.pkl',directory/'model.pkl',directory/'metadata.json')


def _new_directory(model_name):
    run_id=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{model_name}_{uuid.uuid4().hex[:8]}"
    directory=RUNS_DIR/run_id
    directory.mkdir(parents=True)
    return run_id,directory


def archive_original():
    """Snapshot the existing fitted model; do not retrain or overwrite it."""
    digest=hashlib.sha256(b''.join((MODELS_DIR/n).read_bytes() for n in ARTIFACTS)).hexdigest()
    for run in list_runs():
        if run.get('original_artifact_hash')==digest: return run['run_id']
    meta=load_json(MODELS_DIR/'metadata.json')
    run_id,directory=_new_directory(meta['model_name'])
    for name in ARTIFACTS: shutil.copy2(MODELS_DIR/name,directory/name)
    train=joblib.load(directory/'y_train.pkl'); test=joblib.load(directory/'y_test.pkl')
    info={'run_id':run_id,'label':'Original tuned '+MODEL_NAMES.get(meta['model_name'],meta['model_name']),
          'model_name':meta['model_name'],'created_at':datetime.now(timezone.utc).isoformat(),
          'trained_at':meta['trained_at'],'seed':meta['random_seed'], 'features':meta['n_selected_features'],
          'sampler':meta['imbalance_strategy'],'threshold':meta['chosen_threshold'],'test_metrics':meta['test_metrics'],
          'cv_metrics':None,'duration_seconds':None,'dataset_signature':dataset_signature(),
          'split_signature':split_signature(train.index,test.index),'original_artifact_hash':digest,
          'protocol':'Snapshot of the existing tuned model. Comparable OOF metrics were not retained; no CV values invented.'}
    probabilities=load_run_from_directory(directory).predict_proba(joblib.load(directory/'X_test_raw.pkl'))
    pd.DataFrame({'sample_id':test.index,'target':test.to_numpy(),'probability':probabilities}).to_csv(directory/'test_predictions.csv',index=False)
    save_json(info,directory/'run.json')  # Completion marker written last.
    return run_id


def load_run_from_directory(directory):
    return PredictionPipeline.load(directory/'preprocessing_pipeline.pkl',directory/'model.pkl',directory/'metadata.json')


def train_run(model_name, seed=42, n_features=40, sampler='smoteenn', label=''):
    from src.data_loader import load_secom_dataset
    from src.train import get_candidate_models, build_full_pipeline, build_preprocessing_and_selection_pipeline, extract_fitted_preprocessing_pipeline
    from src.evaluate import compute_metrics, find_best_threshold, find_threshold_for_recall
    if model_name not in MODEL_NAMES: raise ValueError('Unsupported model')
    if sampler not in {'none','smote','smoteenn'}: raise ValueError('Unsupported sampler')
    if n_features not in {10,20,40,60}: raise ValueError('Choose 10, 20, 40 or 60 features')
    if not 0 <= seed <= 2**32-1: raise ValueError('Seed must be between 0 and 2^32−1')
    started=time.perf_counter()
    df=load_secom_dataset()
    template=PredictionPipeline.load()
    train_ids=joblib.load(MODELS_DIR/'y_train.pkl').index
    test_ids=joblib.load(MODELS_DIR/'y_test.pkl').index
    if not set(train_ids).isdisjoint(test_ids): raise ValueError('Training and test IDs overlap')
    raw_train=df.loc[train_ids,template.raw_feature_names]; raw_test=df.loc[test_ids,template.raw_feature_names]
    y_train=df.loc[train_ids,'Failure']; y_test=df.loc[test_ids,'Failure']
    classifier=get_candidate_models(seed)[model_name]
    preprocessing=build_preprocessing_and_selection_pipeline(n_features=n_features,random_state=seed)
    full=build_full_pipeline(preprocessing,classifier,sampler_name=sampler,random_state=seed)
    # Fix fold assignment as well as test split across runs; seed varies training only.
    oof=cross_val_predict(full,raw_train,y_train,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba',n_jobs=1)[:,1]
    threshold=find_best_threshold(y_train.to_numpy(),oof)
    cv_metrics=compute_metrics(y_train,(oof>=.5).astype(int),oof)
    full.fit(raw_train,y_train)
    fitted=extract_fitted_preprocessing_pipeline(full)
    model=full.named_steps['classifier']; transformed=fitted.transform(raw_train)
    probabilities=model.predict_proba(fitted.transform(raw_test))[:,1]
    metrics=compute_metrics(y_test,(probabilities>=threshold).astype(int),probabilities)
    tn,fp,fn,tp=confusion_matrix(y_test,(probabilities>=threshold).astype(int),labels=[0,1]).ravel()
    now=datetime.now(timezone.utc).isoformat()
    meta={'model_name':model_name,'model_class':type(model).__name__,'trained_at':now,'random_seed':seed,
          'raw_feature_names':template.raw_feature_names,'selected_feature_names':list(transformed.columns),
          'chosen_threshold':threshold,'best_f1_threshold':threshold,
          'high_recall_threshold':find_threshold_for_recall(y_train.to_numpy(),oof),
          'threshold_source':'training-only out-of-fold predictions (5 folds)',
          'imbalance_strategy':sampler,'hyperparameters':model.get_params(),'test_metrics':metrics,
          'confusion_matrix':dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)),
          'risk_thresholds':DEFAULT_RISK_THRESHOLDS,'n_train':len(raw_train),'n_test':len(raw_test),
          'dataset_failure_rate':float(df.Failure.mean()),'n_raw_features':raw_train.shape[1],
          'n_selected_features':transformed.shape[1]}
    run_id,directory=_new_directory(model_name)
    for name,value in [('model',model),('preprocessing_pipeline',fitted),('X_train_transformed',transformed),
                       ('y_train',y_train),('X_test_raw',raw_test),('y_test',y_test)]: joblib.dump(value,directory/f'{name}.pkl')
    # XGBoost defaults contain NaN placeholders; metadata must be strict JSON.
    meta['hyperparameters']={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in meta['hyperparameters'].items()}
    save_json(meta,directory/'metadata.json')
    pd.DataFrame({'sample_id':test_ids,'target':y_test.to_numpy(),'probability':probabilities}).to_csv(directory/'test_predictions.csv',index=False)
    pd.DataFrame({'sample_id':train_ids,'target':y_train.to_numpy(),'probability':oof}).to_csv(directory/'oof_predictions.csv',index=False)
    info={'run_id':run_id,'label':label.strip() or f'{MODEL_NAMES[model_name]} · seed {seed}',
          'model_name':model_name,'created_at':now,'trained_at':now,'seed':seed,'features':n_features,
          'sampler':sampler,'threshold':threshold,'test_metrics':metrics,'cv_metrics':cv_metrics,
          'duration_seconds':time.perf_counter()-started,'dataset_signature':dataset_signature(),
          'split_signature':split_signature(train_ids,test_ids),
          'versions':{n:importlib.metadata.version(n) for n in ['scikit-learn','xgboost','numpy','pandas','imbalanced-learn']},
          'protocol':'Untuned candidate defaults. Same raw train/test IDs and five CV folds across runs; preprocessing and resampling refit inside every fold. Pooled OOF metrics at 0.5; test metrics at training-OOF-selected threshold. Do not choose models by repeatedly maximizing test scores.'}
    save_json(info,directory/'run.json')
    return run_id
