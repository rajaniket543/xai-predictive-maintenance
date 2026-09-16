"""Run Streamlit pages with a real selected sample and exercise forms/buttons."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

SETUP=f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(ROOT/'app')!r})
import streamlit as st
import state
state.init_session_state()
p = state.require_pipeline()
st.session_state.decision_threshold = p.default_threshold
raw, y = state.get_test_artifacts()
report = state.get_evaluation()
idx = report['demo_sample_id']
row = raw.loc[[idx]]
state.set_current_instance(row, p.transform(row), p.predict(row).iloc[0].to_dict(), 'ui test')
'''

for module, function in [('dashboard','render'),('model_runs','render'),('model_runs','comparison'),('dataset','render'),('performance','render'),('predict','render'),('shap_view','render'),('lime_view','render'),('counterfactual_view','render'),('evaluation','render'),('evaluation','robustness'),('evaluation','analysis'),('evaluation','decision'),('about','render')]:
    app=AppTest.from_string(SETUP+f'\nfrom views import {module}\n{module}.{function}()').run(timeout=120)
    assert not app.exception,[(e.message,e.stack_trace) for e in app.exception]
    assert not app.error,[e.value for e in app.error]
    if function=='decision':
        for label in ['Apply what-if and re-run prediction','Run perturbation validation','Generate model-valid alternatives']:
            next(button for button in app.button if button.label==label).click().run(timeout=120)
            assert not app.exception,[e.message for e in app.exception]
            assert not app.error,[e.value for e in app.error]
    print(f'PASS {module}.{function}',flush=True)
app=AppTest.from_file(str(ROOT/'app/app.py')).run(timeout=120)
assert not app.exception,[e.message for e in app.exception]
print('PASS app entry point and navigation registration')
