"""Experimental evidence and its limits, rendered from versioned computed reports."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import state
from components import page_header, contributor_bar_chart
from src.xai_evaluation import faithfulness, similarities

FAIRNESS = 'Protected demographic fairness could not be directly evaluated because the dataset does not provide appropriate protected-group attributes.'


def report():
    result = state.get_evaluation()
    if result is None:
        st.info('Not evaluated — report is missing or does not match current model/data/code. Compute this run’s XAI results on Model Runs (or run the experiment CLI) and reload.')
        st.stop()
    st.caption(f"Executed {result['created_at']} · seed {result['seed']} · {result['sample_count']} randomly sampled test cases · report threshold {result['threshold']:.2f}")
    return result


def chart(fig):
    fig.update_layout(template='plotly_white', margin=dict(l=20,r=20,t=55,b=35), height=390,
                      colorway=['#2563eb','#0d9488','#ea580c','#7c3aed'])
    st.plotly_chart(fig, width="stretch")


def render():
    page_header('XAI Evaluation', 'Measured evidence about explanation quality — including disagreement and weak results.')
    result=report(); summary=result['summary']
    for column,key in zip(st.columns(5),['faithfulness','fidelity','stability','consistency','robustness']):
        column.metric(key.title(),f"{summary[key]:.3f}")
    st.caption('Faithfulness: excess absolute probability change vs random control. Fidelity: LIME weighted training-neighborhood R². Stability: SHAP cosine at 1% SD noise. Consistency: SHAP/LIME top-5 Jaccard. Robustness: SHAP cosine at 10% SD noise. These are distinct measures, not a combined trust score.')
    tabs=st.tabs(['Faithfulness','Fidelity & consistency','Confidence analysis','Evaluation framework'])
    frame=pd.DataFrame(result['samples'])
    with tabs[0]:
        melted=frame.melt(id_vars='sample_id',value_vars=['explained_change','random_change'],var_name='Perturbation',value_name='Absolute probability change')
        chart(px.bar(melted,x='sample_id',y='Absolute probability change',color='Perturbation',barmode='group',title='Do SHAP-ranked features affect predictions more than random features?',labels={'sample_id':'Test sample ID'}))
        st.caption('Top-5 absolute SHAP features replaced by training medians, compared with 30 random sets of five features per sample. Random-control effects are averaged after taking absolute differences. Negative scores are retained. Marginal replacement can create off-distribution combinations; this is not a causal test.')
        st.dataframe(frame[['sample_id','original_probability','explained_probability','random_probability','explained_change','random_change','faithfulness']],hide_index=True,width="stretch")
    with tabs[1]:
        chart(px.scatter(frame,x='original_probability',y='surrogate_probability',color='fidelity_r2',hover_data=['sample_id','consistency_jaccard'],
                         title='Original model vs LIME surrogate at each explained sample',labels={'original_probability':'Model failure probability','surrogate_probability':'LIME surrogate output','fidelity_r2':'Weighted R²'}))
        st.caption('R² is the distance-weighted fit on the neighborhood used to train LIME, not independent held-out fidelity. Linear outputs can lie outside [0,1]. SHAP explains model output; LIME fits discretized local conditions. Rankings can legitimately disagree.')
        st.dataframe(frame[['sample_id','fidelity_r2','consistency_jaccard','consistency_overlap','consistency_rank_correlation']],hide_index=True,width="stretch")
        comparison()
    with tabs[2]:
        groups=pd.DataFrame(result['confidence_groups'])
        st.dataframe(groups,hide_index=True,width="stretch")
        chart(px.scatter(frame,x='confidence',y='stability',color='faithfulness',hover_data=['sample_id'],title='Does higher confidence imply more reliable explanations?',labels={'confidence':'Uncalibrated max(p, 1−p)','stability':'SHAP cosine similarity at 1% noise'}))
        st.caption('Bins are fixed in advance: low [0.5,0.7), medium [0.7,0.9), high [0.9,1]. Missing bins have no sampled observations. Inspect counts; small groups cannot establish a general relationship. Confidence is not calibration or trustworthiness.')
    with tabs[3]:
        metrics=state.require_pipeline().metadata['test_metrics']
        rows=[('Predictive performance',str({k:round(v,4) for k,v in metrics.items()})),
              ('Interpretability',f"TreeSHAP + LIME executed on {len(frame)} samples"),
              ('Faithfulness',f"Mean excess |Δp|: {summary['faithfulness']:.4f}"),
              ('Fidelity',f"LIME weighted neighborhood R²: {summary['fidelity']:.4f}"),
              ('Stability',f"Mean cosine at 1%: {summary['stability']:.4f}"),
              ('Consistency',f"Mean top-5 Jaccard: {summary['consistency']:.4f}"),
              ('Robustness',f"Mean cosine at 10%: {summary['robustness']:.4f}"),
              ('Fairness','Not evaluated — protected attributes unavailable'),
              ('Confidence',f"{len(groups) if 'groups' in locals() else len(result['confidence_groups'])} observed confidence groups; see computed table"),
              ('Actionability',f"{len(result['counterfactual_demo']['counterfactuals'])} model-valid DiCE alternatives in saved demo; physical actionability unknown"),
              ('Human usefulness','Decision-support workflow implemented; not evaluated in a human study'),
              ('Domain impact','Feature selection and ablation measured; real maintenance impact not evaluated')]
        st.dataframe(pd.DataFrame(rows,columns=['Dimension','Actual evaluation / limitation']),hide_index=True,width="stretch")
        st.info(FAIRNESS)
        st.download_button('Download experimental evidence (JSON)',data=__import__('json').dumps(result,indent=2),file_name='xai_evaluation.json',mime='application/json')


def comparison():
    row=st.session_state.get('selected_transformed_row')
    if row is None:
        st.info('Select a sample on Prediction to compare SHAP and LIME side by side.')
        return
    token=state.artifact_fingerprint(); shap=state.local_shap(token,row); lime=state.local_lime(token,row)
    a=pd.Series(shap['all_shap_values']); b=pd.Series({c['feature']:c['contribution'] for c in lime['contributors']}).reindex(a.index,fill_value=0)
    scores=similarities(a.to_numpy(),b.to_numpy())
    table=pd.DataFrame({f"SHAP {state.shap_units()}":a,'LIME local coefficient':b,'SHAP rank':a.abs().rank(ascending=False,method='min'),
                        'LIME rank':b.abs().rank(ascending=False,method='min')})
    table['In both top-5']=(table['SHAP rank']<=5)&(table['LIME rank']<=5)
    st.subheader('SHAP vs LIME · current sample')
    st.metric('Top-5 Jaccard agreement',f"{scores['jaccard']:.3f}")
    st.dataframe(table.sort_values('SHAP rank').head(20),width="stretch")
    st.caption('Unselected LIME features are zero-padded for ranking comparison. Coefficient magnitudes use different units and must not be equated.')


def robustness():
    page_header('Robustness', 'Separate prediction stability from explanation stability under controlled input noise.')
    result=report(); frame=pd.DataFrame(result['robustness'])
    avg=frame.groupby('perturbation',as_index=False)[['cosine','top_k_overlap','prediction_stable','probability_change']].mean()
    avg['Noise (% of training SD)']=avg.perturbation*100
    left,right=st.columns(2)
    with left:
        melted=avg.melt(id_vars='Noise (% of training SD)',value_vars=['cosine','top_k_overlap'],var_name='Measure',value_name='Similarity')
        chart(px.line(melted,x='Noise (% of training SD)',y='Similarity',color='Measure',markers=True,title='Explanation robustness'))
    with right:
        chart(px.line(avg,x='Noise (% of training SD)',y='prediction_stable',markers=True,title='Model robustness: unchanged predicted class',labels={'prediction_stable':'Fraction of unchanged predictions'}))
    st.dataframe(avg,hide_index=True,width="stretch")
    st.caption('Gaussian noise scaled by each feature’s training standard deviation; same random direction at all magnitudes. No clipping (0% is an exact identity control). Mean scores over sampled cases, one noise draw per case; no guarantee of real-world robustness.')
    current = st.session_state.get('selected_transformed_row')
    if current is not None:
        with st.expander('Live stability check · current prediction sample', expanded=True):
            magnitude = st.slider('Noise magnitude (% of training SD)', 0, 10, 1)
            token = state.artifact_fingerprint()
            original = state.local_shap(token, current)
            training, _ = state.get_training_artifacts()
            noise = np.random.default_rng(42).normal(size=current.shape) * training.std().to_numpy()
            changed = pd.DataFrame(current.to_numpy() + magnitude / 100 * noise, columns=current.columns, index=current.index)
            perturbed = state.local_shap(token, changed)
            a = np.array(list(original['all_shap_values'].values()))
            b = np.array(list(perturbed['all_shap_values'].values()))
            sim = similarities(a, b)
            cols = st.columns(3)
            cols[0].metric('Explanation cosine', f"{sim['cosine']:.4f}" if sim['cosine'] is not None else 'Undefined (zero vector)')
            cols[1].metric('Original probability', f"{original['failure_probability']:.1%}")
            cols[2].metric('Perturbed probability', f"{perturbed['failure_probability']:.1%}")
            top = np.argsort(-np.abs(a))[:10]
            data = pd.DataFrame({'feature': current.columns[top], 'Original': a[top], 'Perturbed': b[top]})
            chart(px.bar(data.melt(id_vars='feature', var_name='Input', value_name='SHAP contribution'), x='feature', y='SHAP contribution', color='Input', barmode='group', title='Current prediction: original vs perturbed SHAP'))
    selected=st.selectbox('Inspect explanation stability for sample',options=[v['sample_id'] for v in result['stability_vectors']])
    vector=next(v for v in result['stability_vectors'] if v['sample_id']==selected)
    pairs=pd.DataFrame({'feature':result['features'],'Original':vector['original'],'Perturbed (1%)':vector['perturbed']})
    pairs=pairs.iloc[np.argsort(-np.abs(pairs.Original.to_numpy()))[:10]]
    chart(px.bar(pairs.melt(id_vars='feature',var_name='Input',value_name=f"SHAP {state.shap_units()}"),x='feature',y=f"SHAP {state.shap_units()}",color='Input',barmode='group',title='Same sample: original and perturbed explanation'))


def analysis():
    page_header('Explanation-Guided Analysis','Use explanations to form hypotheses, retrain, and measure the outcome.')
    result=report(); experiment=result['feature_analysis']
    st.subheader('All features → training SHAP ranking → Top-K / ablation → retrain → test')
    frame=pd.DataFrame(experiment['results'])
    st.dataframe(frame,hide_index=True,width="stretch")
    chart(px.bar(frame.melt(id_vars='configuration',value_vars=['f1','recall','roc_auc','pr_auc'],var_name='Metric',value_name='Score'),x='Metric',y='Score',color='configuration',barmode='group',title='Does explanation-guided feature selection help held-out performance?'))
    st.caption(experiment['protocol'])
    st.info('Potentially spurious dependence is a hypothesis, not a finding. A performance change after ablation shows model dependence, not that a sensor is physically irrelevant or causally spurious.')
    with st.expander('Selected and ablated features'):
        st.write('SHAP Top-40:',experiment['selected_features']); st.write('Removed Top-5:',experiment['ablated_features'])
    st.subheader('Controlled distribution-shift experiment')
    shift=result['distribution_shift']
    st.dataframe(pd.DataFrame(shift['results']),hide_index=True,width="stretch")
    st.metric('Mean original / shifted SHAP cosine',f"{shift['explanation_similarity']:.3f}")
    importance=pd.DataFrame(shift['importance']).nlargest(12,'original')
    chart(px.bar(importance.melt(id_vars='feature',var_name='Distribution',value_name='Mean absolute SHAP'),x='feature',y='Mean absolute SHAP',color='Distribution',barmode='group',title='Which feature importances change under the synthetic shift?'))
    st.caption(shift['protocol'])


def decision():
    pipeline=state.require_pipeline()
    page_header('Decision Support','Prediction → primary factors → what-if → validation → maintenance investigation')
    row=st.session_state.get('selected_transformed_row')
    if row is None:
        st.info('Select and predict a test sample on Prediction to start the investigation.'); return
    from components import render_prediction_result
    render_prediction_result(st.session_state.last_prediction)
    token=state.artifact_fingerprint(); explanation=state.local_shap(token,row)
    contributions=explanation['top_contributors']
    st.subheader('Primary factors')
    chart(contributor_bar_chart([c['feature'] for c in contributions[:6]],[c['shap_value'] for c in contributions[:6]],'Why did the model flag this run?',f"SHAP contribution ({state.shap_units()})"))
    training,_=state.get_training_artifacts()
    feature=st.selectbox('Influential feature to investigate',options=[c['feature'] for c in contributions])
    original=float(row[feature].iloc[0]); median=float(training[feature].median())
    with st.form('intervention'):
        value=st.number_input('Replacement value (standardized model units)',value=median,format='%.5f')
        st.caption(f'Original {original:.5f} · training median {median:.5f}. This edits model input, not a physical machine setting.')
        submitted=st.form_submit_button('Apply what-if and re-run prediction',type='primary')
    if submitted:
        if not np.isfinite(value): st.error('Enter a finite value.'); return
        changed=row.copy(); changed[feature]=value
        p0=explanation['failure_probability']; p1=float(pipeline.model.predict_proba(changed)[0,1])
        st.session_state['intervention_result']={'key':(token,row.to_json(),feature),'before':p0,'after':p1,'value':value}
    intervention=st.session_state.get('intervention_result')
    if intervention and intervention['key']==(token,row.to_json(),feature):
        left,right=st.columns(2); left.metric('Original failure probability',f"{intervention['before']:.1%}")
        right.metric('After model-level intervention',f"{intervention['after']:.1%}",f"{100*(intervention['after']-intervention['before']):+.2f} percentage points",delta_color='inverse')
        st.caption(f"{feature}: {original:.5f} → {intervention['value']:.5f}. This demonstrates model sensitivity, not a causal maintenance benefit.")
    with st.expander('Validate SHAP-ranked features against a random control'):
        if st.button('Run perturbation validation'):
            values=np.array([explanation['all_shap_values'][f] for f in row.columns])
            measured=faithfulness(pipeline.model,row,values,training)
            chart(px.bar(x=['Original','Top-5 replaced','Random-5 mean'],y=[measured['original_probability'],measured['explained_probability'],measured['random_probability']],labels={'x':'Input','y':'Failure probability'},title='Re-evaluation after median replacement'))
            st.metric('Faithfulness: excess absolute probability change',f"{measured['faithfulness']:.4f}")
    with st.expander('What-if alternatives · DiCE'):
        if st.button('Generate model-valid alternatives'):
            cf=state.counterfactuals(token,row,2,state.global_importance(token).feature.head(15).tolist(),st.session_state.decision_threshold)
            if cf['counterfactuals']:
                for candidate in cf['counterfactuals']:
                    st.write(f"Failure probability: {candidate['failure_probability']:.1%}")
                    st.dataframe(pd.DataFrame(candidate['changes']),hide_index=True)
            else: st.info('No counterfactual found within this search budget and the permitted ranges.')
    st.subheader('Maintenance insight')
    st.write(f"Review the measurements and data quality associated with {', '.join(c['feature'] for c in contributions[:3])}. These anonymous variables influenced the model; their physical sensor meanings and adjustability are unknown. Ask a process engineer to map them to equipment records before considering maintenance action. Check missed failures and explanation reliability alongside this score.")
    st.caption('Human usefulness and real operational impact have not been measured. This workflow supports investigation; it does not prescribe machine adjustments.')
