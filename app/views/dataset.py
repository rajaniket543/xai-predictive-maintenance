import pandas as pd
import plotly.express as px
import streamlit as st
import state
from components import page_header
from src.data_loader import load_secom_dataset, get_feature_columns
from views.evaluation import FAIRNESS, chart

@st.cache_data
def load_data(token):
    return load_secom_dataset()


def render():
    page_header('Dataset', 'SECOM · manufacturing run outcomes and anonymous process measurements')
    try: df=load_data(state.artifact_fingerprint())
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc)); return
    features=get_feature_columns(df)
    for col,label,value in zip(st.columns(4),['Runs','Raw sensors','Failure rate','Missing cells'],[f'{len(df):,}',str(len(features)),f"{df.Failure.mean():.2%}",f'{df[features].isna().to_numpy().mean():.2%}']):
        col.metric(label,value)
    left,right=st.columns(2)
    with left:
        counts=df.Failure.map({0:'Normal',1:'Failure'}).value_counts().rename_axis('Outcome').reset_index(name='Runs')
        chart(px.bar(counts,x='Outcome',y='Runs',color='Outcome',title='Why accuracy alone is insufficient'))
    with right:
        missing=df[features].isna().mean().nlargest(15).rename_axis('Sensor').reset_index(name='Missing fraction')
        chart(px.bar(missing,x='Missing fraction',y='Sensor',orientation='h',title='Where are sensor measurements missing?'))
    st.write('Training-only preprocessing: drop >55% missing columns → median imputation → remove constants → standard scaling → correlation filtering → mutual-information Top-40. Sampling for class imbalance occurs only inside training folds.')
    st.info(FAIRNESS)
    st.caption('Timestamps describe historical runs. This project uses a stratified random split, not prospective temporal validation. Sensors have no verified physical meanings.')
    with st.expander('Inspect raw data (missing values preserved)'):
        st.dataframe(df.head(50),width="stretch")
