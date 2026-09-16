"""Integration checks against real saved data/model; no fabricated model metrics."""
import unittest
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from src.prediction import PredictionPipeline, InvalidInputError
from src.utils import MODELS_DIR, METRICS_DIR
from src.explain_shap import SHAPExplainer
from src.explain_lime import LIMEExplainer
from src.explain_dice import DiCEExplainer
from src.xai_evaluation import similarities, faithfulness, artifact_fingerprint


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline=PredictionPipeline.load()
        cls.train=joblib.load(MODELS_DIR/'X_train_transformed.pkl')
        cls.y_train=joblib.load(MODELS_DIR/'y_train.pkl')
        cls.raw=joblib.load(MODELS_DIR/'X_test_raw.pkl')
        cls.test=cls.pipeline.transform(cls.raw)
        cls.shap=SHAPExplainer(cls.pipeline.model,cls.train)

    def test_split_and_preprocessing(self):
        self.assertTrue(set(self.train.index).isdisjoint(self.test.index))
        self.assertEqual(list(self.test.columns),self.pipeline.selected_feature_names)
        self.assertTrue(np.isfinite(self.test.to_numpy()).all())
        np.testing.assert_allclose(self.pipeline.predict_proba(self.raw),self.pipeline.model.predict_proba(self.test)[:,1])

    def test_invalid_input(self):
        for bad in [self.raw.iloc[:0],self.raw.iloc[:1].drop(columns=self.raw.columns[0]),self.raw.iloc[:1].assign(Feature_0=np.inf),self.raw.iloc[:1].assign(Feature_0='bad')]:
            with self.assertRaises(InvalidInputError): self.pipeline.predict_proba(bad)

    def test_shap_additivity(self):
        row=self.test.iloc[[0]]; result=self.shap.local_explanation(row)
        reconstructed=expit(result['base_value']+sum(result['all_shap_values'].values()))
        self.assertAlmostEqual(reconstructed,result['failure_probability'],places=5)

    def test_lime_repeatability_and_fidelity(self):
        lime=LIMEExplainer(self.train)
        a=lime.explain_instance(self.pipeline.model,self.test.iloc[[0]])
        b=lime.explain_instance(self.pipeline.model,self.test.iloc[[0]])
        self.assertEqual(a['contributors'],b['contributors'])
        self.assertTrue(np.isfinite(a['local_model_r2']))
        self.assertTrue(np.isfinite(a['surrogate_probability']))

    def test_faithfulness_matches_direct_prediction(self):
        row=self.test.iloc[[0]]; values=list(self.shap.local_explanation(row)['all_shap_values'].values())
        result=faithfulness(self.pipeline.model,row,values,self.train)
        changed=row.copy(); changed[result['features']]=self.train[result['features']].median().values
        self.assertAlmostEqual(result['explained_probability'],float(self.pipeline.model.predict_proba(changed)[0,1]),places=6)
        self.assertAlmostEqual(result['faithfulness'],result['explained_change']-result['random_change'],places=6)
        self.assertEqual(result,faithfulness(self.pipeline.model,row,values,self.train))

    def test_similarity_controls(self):
        a=np.array([1.,-3.,2.,4.,.5,0.])
        result=similarities(a,a)
        self.assertAlmostEqual(result['cosine'],1)
        self.assertEqual(result['jaccard'],1)
        self.assertEqual(result['top_k_overlap'],1)
        self.assertIsNone(similarities(np.zeros(6),np.zeros(6))['cosine'])

    def test_experiment_integrity(self):
        report=json.loads((METRICS_DIR/'xai_evaluation.json').read_text())
        self.assertEqual(report['fingerprint'],artifact_fingerprint())
        self.assertIn('training-only',self.pipeline.metadata['threshold_source'])
        zero=[r for r in report['robustness'] if r['perturbation']==0]
        self.assertTrue(all(r['prediction_stable']==1 and abs(r['cosine']-1)<1e-6 and r['probability_change']==0 for r in zero))
        analysis=report['feature_analysis']
        self.assertTrue(set(analysis['training_ids']).isdisjoint(analysis['test_ids']))
        self.assertEqual(analysis['results'][1]['features'],40)
        self.assertEqual(analysis['results'][0]['features']-analysis['results'][2]['features'],5)
        for key,column in [('faithfulness','faithfulness'),('fidelity','fidelity_r2'),('stability','stability'),('consistency','consistency_jaccard')]:
            self.assertAlmostEqual(report['summary'][key],np.mean([r[column] for r in report['samples']]))

    def test_counterfactual_validation(self):
        report=json.loads((METRICS_DIR/'xai_evaluation.json').read_text())
        row=self.test.loc[[report['demo_sample_id']]]
        cf=report['counterfactual_demo']
        self.assertTrue(cf['counterfactuals'],'Demo should contain a verified candidate')
        for candidate in cf['counterfactuals']:
            modified=row.copy()
            for change in candidate['changes']: modified[change['feature']]=change['counterfactual_value']
            p=float(self.pipeline.model.predict_proba(modified)[0,1])
            self.assertAlmostEqual(p,candidate['failure_probability'],places=6)
            self.assertNotEqual(p>=report['threshold'],cf['original_probability']>=report['threshold'])

if __name__=='__main__': unittest.main()
