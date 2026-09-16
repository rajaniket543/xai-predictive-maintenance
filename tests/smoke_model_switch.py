"""Exercise session-scoped switching and cache isolation through Streamlit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from src.model_runs import list_runs
runs=list_runs()
script=f'''
import sys
sys.path.insert(0,{str(ROOT)!r})
sys.path.insert(0,{str(ROOT/'app')!r})
import streamlit as st
import state
from src.model_runs import list_runs
state.init_session_state()
if st.session_state.decision_threshold is None:
    st.session_state.decision_threshold=state.require_pipeline().default_threshold
if st.session_state.selected_raw_row is None:
    p=state.require_pipeline()
    raw,_=state.get_test_artifacts()
    row=raw.loc[[238]]
    state.set_current_instance(row,p.transform(row),p.predict(row).iloc[0].to_dict(),'test')
for r in list_runs():
    st.button(r['run_id'],on_click=state.activate_run,args=(r['run_id'],))
st.write(type(state.require_pipeline().model).__name__)
from views import predict, shap_view, lime_view, evaluation
predict.render()
shap_view.render()
lime_view.render()
'''
app=AppTest.from_string(script).run(timeout=120)
assert not app.exception,[e.message for e in app.exception]
for run in runs:
    next(b for b in app.button if b.label==run['run_id']).click().run(timeout=120)
    assert not app.exception,[e.message for e in app.exception]
    assert not app.error,[e.value for e in app.error]
    assert app.session_state.active_run_id==run['run_id']
    assert app.session_state.decision_threshold==run['threshold']
    assert int(app.session_state.selected_raw_row.index[0])==238
    assert app.session_state.last_prediction['threshold_used']==run['threshold']
    assert app.session_state.prediction_history[-1]['run_id']==run['run_id']
    print('PASS activation and SHAP/LIME cache isolation:',run['label'],flush=True)
