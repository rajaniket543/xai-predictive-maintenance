"""Validate saved model comparisons and run isolation with real artifacts."""
import unittest
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from src.model_runs import list_runs, run_directory, load_run, archive_original, train_run
from src.utils import MODELS_DIR
from src.evaluate import compute_metrics, find_best_threshold
from src.explain_shap import SHAPExplainer
from src.xai_evaluation import artifact_fingerprint


class ModelRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs=list_runs()

    def test_actual_metrics_match_saved_predictions(self):
        self.assertGreaterEqual(len(self.runs),3)
        for run in self.runs:
            directory=run_directory(run['run_id']); pipeline=load_run(run['run_id'])
            raw=joblib.load(directory/'X_test_raw.pkl'); target=joblib.load(directory/'y_test.pkl')
            p=pipeline.predict_proba(raw)
            saved=pd.read_csv(directory/'test_predictions.csv')
            np.testing.assert_allclose(p,saved.probability,rtol=1e-6)
            self.assertEqual(list(raw.index),list(saved.sample_id))
            metrics=compute_metrics(target,(p>=pipeline.default_threshold).astype(int),p)
            for name,value in metrics.items(): self.assertAlmostEqual(value,run['test_metrics'][name],places=7)
            oof=directory/'oof_predictions.csv'
            if oof.exists():
                frame=pd.read_csv(oof)
                self.assertTrue(set(frame.sample_id).isdisjoint(saved.sample_id))
                self.assertAlmostEqual(find_best_threshold(frame.target,frame.probability),run['threshold'])

    def test_same_comparison_cohort_and_separate_artifacts(self):
        self.assertEqual(len({r['split_signature'] for r in self.runs}),1)
        self.assertEqual(len({r['dataset_signature'] for r in self.runs}),1)
        tokens=[artifact_fingerprint(run_directory(r['run_id'])) for r in self.runs]
        self.assertEqual(len(tokens),len(set(tokens)))

    def test_snapshot_is_idempotent_and_preserves_original(self):
        before=hashlib.sha256((MODELS_DIR/'model.pkl').read_bytes()).hexdigest()
        run_id=archive_original()
        self.assertEqual(archive_original(),run_id)
        self.assertEqual(before,hashlib.sha256((run_directory(run_id)/'model.pkl').read_bytes()).hexdigest())
        self.assertEqual(before,hashlib.sha256((MODELS_DIR/'model.pkl').read_bytes()).hexdigest())

    def test_different_model_explanations_use_correct_units(self):
        for run in self.runs:
            directory=run_directory(run['run_id']); pipeline=load_run(run['run_id'])
            training=joblib.load(directory/'X_train_transformed.pkl')
            raw=joblib.load(directory/'X_test_raw.pkl').iloc[[0]]
            explained=SHAPExplainer(pipeline.model,training).local_explanation(pipeline.transform(raw))
            total=explained['base_value']+sum(explained['all_shap_values'].values())
            if explained['output_units']=='log-odds': total=1/(1+np.exp(-total))
            self.assertAlmostEqual(total,explained['failure_probability'],places=5)

    def test_invalid_run_ids_and_configuration(self):
        for name in ['../metadata.json','/tmp','.hidden','missing']:
            with self.assertRaises(ValueError): run_directory(name)
        with self.assertRaises(ValueError): train_run('unsupported')

if __name__=='__main__': unittest.main()
