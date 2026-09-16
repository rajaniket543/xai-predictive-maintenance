"""Train, activate, and compare separately saved model runs."""
import json
import subprocess
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import state
from components import page_header
from src.model_runs import MODEL_NAMES, list_runs, train_run, archive_original, run_directory
from src.utils import PROJECT_ROOT, load_json
from src.xai_evaluation import artifact_fingerprint


def render():
    page_header('Model Runs', 'Train another algorithm, keep its history, and activate it across the application.')
    pipeline=state.require_pipeline()
    st.info(f"Active model: {pipeline.metadata['model_class']} · run: {state.active_run_id()} · default threshold: {pipeline.default_threshold:.2f}")
    st.caption('Switching a saved run changes predictions, preprocessing, SHAP, LIME, DiCE and evaluation results together. The selected raw sample is retained and re-scored. Original XGBoost files are preserved.')
    with st.form('train_model_run'):
        left,right=st.columns(2)
        with left:
            model=st.selectbox('Algorithm',options=list(MODEL_NAMES),format_func=MODEL_NAMES.get)
            features=st.selectbox('Selected features',options=[10,20,40,60],index=2)
            label=st.text_input('Run label (optional)',max_chars=100)
        with right:
            seed=st.number_input('Training seed',min_value=0,max_value=2**32-1,value=42,step=1)
            sampler=st.selectbox('Imbalance handling',options=['smoteenn','smote','none'])
        st.caption('Candidate defaults, without hyperparameter search. Same train/test IDs and five CV folds for every run. The seed changes model training, feature selection and resampling. Threshold is chosen from training-only out-of-fold predictions.')
        submitted=st.form_submit_button('Train and save run',type='primary')
    if submitted:
        try:
            with st.spinner('Training five folds, selecting a threshold, and evaluating the saved model…'):
                archive_original()
                run_id=train_run(model,int(seed),features,sampler,label)
            st.success(f'Saved {run_id}. Select it below or use Active model run in the sidebar.')
        except Exception as exc:
            st.error(f'Training failed; the active model was not replaced. {exc}')
    runs=list_runs()
    if not runs:
        st.info('No saved runs yet. Train one, or archive the existing fitted model.')
        if st.button('Save original model as a run'):
            archive_original(); st.rerun()
        return
    lookup={r['run_id']:r for r in runs}
    selected=st.selectbox('Inspect saved run',options=list(lookup),format_func=lambda key:f"{lookup[key]['label']} · {key}")
    chosen=lookup[selected]
    columns=st.columns(4)
    for col,key in zip(columns,['f1','recall','roc_auc','pr_auc']): col.metric(key.upper(),f"{chosen['test_metrics'][key]:.3f}")
    st.button('Use this model throughout the app',on_click=state.activate_run,args=(selected,),type='primary')
    st.caption('Activation is local to this browser session. Saved runs remain available after restarting the app; a new session starts with the original working model.')
    with st.expander('Run configuration and provenance'):
        st.json(chosen)
        st.json(load_json(run_directory(selected)/'metadata.json')['hyperparameters'])
    st.subheader('XAI experiments for this run')
    st.caption('Live SHAP/LIME/DiCE work immediately. Full evaluation and feature-ablation experiments are computed separately and saved for this specific run.')
    if st.button('Compute XAI evaluation for selected run'):
        with st.spinner('Running the existing experiment suite for the selected model…'):
            result=subprocess.run([sys.executable,str(PROJECT_ROOT/'run_experiments.py'),'--run',selected],cwd=PROJECT_ROOT,capture_output=True,text=True)
        if result.returncode:
            st.error('XAI experiments did not complete. Predictions and saved training results remain available.')
            with st.expander('Execution details'): st.code((result.stdout+result.stderr)[-12000:])
        else:
            st.success('XAI results saved for this run. Activate it to explore its evaluation pages.')
    st.code(f'python run_experiments.py --run {selected}',language='bash')


def comparison():
    page_header('Run Comparison', 'Compare actual saved training runs across algorithms, seeds, feature counts and sampling strategies.')
    runs=list_runs()
    if len(runs)<2:
        st.info('Train at least two runs on Model Runs to compare them here.'); return
    lookup={r['run_id']:r for r in runs}
    selected=st.multiselect('Runs to compare',options=list(lookup),default=list(lookup),format_func=lambda key:f"{lookup[key]['label']} · {key}")
    if not selected:
        st.info('Select one or more runs.'); return
    records=[lookup[k] for k in selected]
    cohorts={(r['dataset_signature'],r['split_signature']) for r in records}
    if len(cohorts)>1:
        st.warning('These runs used different datasets or splits. Select runs from one cohort before comparing scores. See run provenance on Model Runs.'); return
    scope=st.radio('Evaluation scope',['Held-out test','Training out-of-fold (fixed 0.5)'],horizontal=True)
    key='test_metrics' if scope=='Held-out test' else 'cv_metrics'
    rows=[]
    missing=[]
    for r in records:
        metrics=r.get(key)
        if metrics is None:
            missing.append(r['label']); continue
        rows.append({'Run':r['run_id'],'Label':r['label'],'Model':MODEL_NAMES[r['model_name']],
                     'Seed':r['seed'],'Features':r['features'],'Sampler':r['sampler'],
                     'Threshold':r['threshold'] if key=='test_metrics' else .5,
                     'Created':r['created_at'],'Training seconds':r['duration_seconds'],**metrics})
    if missing: st.info('Not evaluated — comparable OOF metrics were not retained for: '+', '.join(missing))
    if not rows: return
    frame=pd.DataFrame(rows)
    st.dataframe(frame,hide_index=True,width='stretch')
    st.download_button('Download run comparison CSV',frame.to_csv(index=False),file_name='model_run_comparison.csv',mime='text/csv')
    st.caption('Same raw train/test IDs and labels. Test metrics use each run’s training-selected threshold; pooled OOF metrics use fixed 0.5. Original tuned XGBoost versus new untuned defaults is not an equal-tuning-budget comparison. Repeated test comparisons are exploratory; avoid selecting models by test scores.')
    metric=st.selectbox('Metrics to plot',options=['f1','recall','pr_auc','roc_auc','precision','accuracy'])
    fig=px.bar(frame,x='Run',y=metric,color='Model',hover_data=['Label','Seed','Features','Sampler','Threshold'],title=f'{metric.upper()} across saved runs')
    fig.update_layout(template='plotly_white',xaxis_title='Saved training run',yaxis_title=metric.upper(),height=430)
    st.plotly_chart(fig,width='stretch')
    st.subheader('XAI quality across runs')
    evidence=[]
    for r in records:
        directory=run_directory(r['run_id']); path=directory/'outputs/metrics/xai_evaluation.json'
        report=load_json(path) if path.exists() else None
        if report and report.get('fingerprint')==artifact_fingerprint(directory):
            evidence.append({'Run':r['run_id'],'Status':'Evaluated','Samples':report['sample_count'],**report['summary']})
        else: evidence.append({'Run':r['run_id'],'Status':'Not evaluated — report missing or stale'})
    st.dataframe(pd.DataFrame(evidence),hide_index=True,width='stretch')
    st.caption('Faithfulness, fidelity, stability, consistency and robustness are separate measures; none is an overall trust percentage. Evaluation uses fixed seed 42 and the same sampled test IDs for the same sample count.')
    if len(records)>=2:
        st.subheader('Same test samples · different model predictions')
        left,right=st.columns(2)
        a=left.selectbox('Run A',selected,format_func=lambda k:lookup[k]['label'])
        b=right.selectbox('Run B',selected,index=1,format_func=lambda k:lookup[k]['label'])
        if a==b:
            st.info('Choose two different runs to inspect disagreement.'); return
        pa=pd.read_csv(run_directory(a)/'test_predictions.csv'); pb=pd.read_csv(run_directory(b)/'test_predictions.csv')
        pair=pa.merge(pb,on=['sample_id','target'],suffixes=('_a','_b'),validate='one_to_one')
        pair['prediction_a']=(pair.probability_a>=lookup[a]['threshold']).astype(int)
        pair['prediction_b']=(pair.probability_b>=lookup[b]['threshold']).astype(int)
        pair['disagree']=pair.prediction_a!=pair.prediction_b
        pair['absolute_probability_difference']=(pair.probability_a-pair.probability_b).abs()
        st.metric('Predicted-class agreement',f"{1-pair.disagree.mean():.1%}")
        fig=px.scatter(pair,x='probability_a',y='probability_b',color='disagree',hover_data=['sample_id','target'],title='Failure probabilities for the same held-out samples',labels={'probability_a':lookup[a]['label'],'probability_b':lookup[b]['label']})
        fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash',color='gray'))
        st.plotly_chart(fig,width='stretch')
        st.dataframe(pair.sort_values('absolute_probability_difference',ascending=False),hide_index=True,width='stretch')
        for col,run in zip(st.columns(2),[lookup[a],lookup[b]]):
            cm=load_json(run_directory(run['run_id'])/'metadata.json')['confusion_matrix']
            with col:
                fig=go.Figure(go.Heatmap(z=[[cm['tn'],cm['fp']],[cm['fn'],cm['tp']]],x=['Predicted normal','Predicted failure'],y=['Actual normal','Actual failure'],texttemplate='%{z}',colorscale='Blues',showscale=False))
                fig.update_layout(title=run['label'],height=330,yaxis_autorange='reversed')
                st.plotly_chart(fig,width='stretch')
