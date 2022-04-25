import streamlit as st
import shap
from streamlit_shap import st_shap
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import json
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import settings as conf

st.set_page_config(
    page_title="Score Credit Dashboard",
    # layout="wide",
    initial_sidebar_state="expanded",
)

# API_BASE_URL = "http://ec2-35-181-58-38.eu-west-3.compute.amazonaws.com:8000"


@st.cache
def load_all_customers_id():
    id_list = requests.get(f"{conf.API_BASE_URL}/customers")
    return id_list.json()


@st.cache
def get_customer_info_df(customer_id):
    customer = requests.get(f"{conf.API_BASE_URL}/detail/{customer_id}")

    return pd.DataFrame(customer.json()['raw']), pd.DataFrame(customer.json()['processed'])


@st.cache
def get_population_df():
    population = requests.get(f"{conf.API_BASE_URL}/population")

    return pd.DataFrame(population.json()['raw_population']).replace('NAN', np.nan)


@st.cache
def get_customer_prediction(customer_data):
    customer_details = customer_data.fillna('NAN').to_dict('list')
    prediction = requests.post(f"{conf.API_BASE_URL}/predict",
                               data=json.dumps({'data': customer_details}))
    print(prediction.json())
    return prediction.json()


@st.cache()
def get_chart_data(feature_name, customer_id):
    return requests.post(f"{conf.API_BASE_URL}/chart",
                         data=json.dumps({'customer_id': customer_id, 'feature': feature_name}))


def get_chart(feature_name, customer_id, max_limit):
    chart_data = get_chart_data(feature_name, customer_id)

    pop_data = {feature_name[0]: chart_data.json()['population_0'],
                'target': chart_data.json()['target'],
                'Type': 'other'}
    cust_data = {feature_name[0]: chart_data.json()['customer_0'],
                 'target': 0,
                 'Type': 'customer'}

    if len(feature_name) > 1:
        pop_data[feature_name[1]] = chart_data.json()['population_1']
        cust_data[feature_name[1]] = chart_data.json()['customer_1']

    ####
    df = pd.DataFrame(pop_data)
    df_customer = pd.DataFrame(cust_data)
    df = df.append(df_customer, ignore_index=True)

    ###
    fig, ax = plt.subplots(figsize=(20, 10))
    if len(feature_name) == 1:
        ax = sns.histplot(data=df[df[feature_name[0]] < max_limit], x=feature_name[0], hue='target', bins=30)
        ax.axvline(df[df['Type'] == 'customer'][feature_name[0]].values, color="red", linestyle=":")
        plt.xlabel(feature_name[0])
        plt.ylabel('Count')

    if len(feature_name) > 1:
        ax = sns.scatterplot(data=df[(df[feature_name[0]] < max_limit) & (df[feature_name[1]] < max_limit)],
                             x=feature_name[0], y=feature_name[1], hue='target', alpha=.5)
        # ax.set_xticklabels(rotation=30)
        ax = sns.scatterplot(data=df[df['Type'] == 'customer'], x=feature_name[0], y=feature_name[1], color='red')
        plt.xlabel(feature_name[0])
        plt.ylabel(feature_name[1])
    return fig


@st.cache()
def load_shap(customer_id):
    model = pickle.load(open(f"{conf.MODEL_PATH}/lgbm_undersample.pkl", 'rb'))
    test_data = pd.read_csv(f"{conf.COMPUTE_DATA_PATH}/transform_df_test.csv")
    explainer = shap.Explainer(model)
    shap_values = explainer(test_data)

    customer_index = test_data.index[test_data['SK_ID_CURR'] == customer_id].tolist()[0]
    return shap_values[customer_index]


# Title.
st.title("Dashboard  intéractif : Détection de défaut de paiement")

# Header & Subheader.
st.header("Projet 7 - Implémenter un modéle de scoring")

# Markdown.
st.markdown(
    "Ce dashboard expose la probabilité de défaut de paiement d'un client. Il permet une visualisation dynamique de certains indicateurs ainsi que le calcul d'une nouvelle probabilité de défaut.")

### Info client et prediction ####
rsk = st.columns([1, 2, 2])
customer_id = rsk[0].selectbox('Saisir l\'identifiant d\'un client:', load_all_customers_id())

customer_df_raw, customer_df = get_customer_info_df(customer_id)
customer_prediction = get_customer_prediction(customer_df)

rsk[1].metric("Score", f"{customer_prediction['proba'][1]:.2f}", delta=None, delta_color="normal")
rsk[2].metric("Risque", customer_prediction['label'], delta=None, delta_color="normal")

g_info = st.columns(4)
g_info[0].metric("revenus", f"{customer_df_raw['AMT_INCOME_TOTAL'][0]:.0f} $", delta=None, delta_color="normal")
g_info[1].metric("montant du bien", f"{customer_df_raw['AMT_GOODS_PRICE'][0]:.0f} $", delta=None, delta_color="normal")
g_info[2].metric("credit", f"{customer_df_raw['AMT_CREDIT'][0]:.0f} $", delta=None, delta_color="normal")
g_info[3].metric("annuitÃ©s", f"{customer_df_raw['AMT_ANNUITY'][0]:.0f} $", delta=None, delta_color="normal")

#### graphiques ####

# SHAP Explanation
shap_values = load_shap(customer_id)
st_shap(shap.plots.waterfall(shap_values), height=500)

# Graph Income/Credit & Days Employed.
if 'population' not in st.session_state:
    st.session_state['population'] = get_population_df()

df = st.session_state['population']


def plot_1():
    income_client = int(customer_df_raw['AMT_INCOME_TOTAL'])
    percent_income_client = income_client * 0.1
    df = get_population_df()

    mask_1 = (df['AMT_INCOME_TOTAL'] <= income_client + percent_income_client)
    mask_2 = (df['AMT_INCOME_TOTAL'] >= income_client - percent_income_client)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Income & Credits", "Days Employed"))

    trace0 = go.Histogram(x=df[mask_1 & mask_2]['AMT_CREDIT'],
                          name='Income & Credit',
                          xbins=dict(size=100000),
                          histnorm='percent',
                          marker_color='#EB89B5')

    trace0_client = go.Scatter(x=[int(customer_df_raw['AMT_CREDIT']), int(customer_df_raw['AMT_CREDIT'])],
                               y=[0, 20], mode="lines", name="Client's credit",
                               line=go.scatter.Line(color="red"))

    trace1 = go.Histogram(x=-df['DAYS_EMPLOYED'],
                          name='Days Employed',
                          xbins=dict(size=500),
                          marker_color='#37AA9C',
                          histnorm='percent')

    trace1_client = go.Scatter(x=[int(customer_df_raw['DAYS_EMPLOYED']), int(customer_df_raw['DAYS_EMPLOYED'])],
                               mode="lines", y=[0, 14.5],
                               line=go.scatter.Line(color="black"), name="Client's days employed")

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 2, 1)
    fig.append_trace(trace0_client, 1, 1)
    fig.append_trace(trace1_client, 2, 1)

    fig.update_layout(height=640, width=850)

    # Update yaxis properties
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="â‚¬", row=1, col=1)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    return fig


st.plotly_chart(plot_1())

# Indicator
st.error('Autres indicateurs :')

mask_target1 = (df['TARGET'] == 1)
mask_target0 = (df['TARGET'] == 0)

data_source1 = [df[mask_target1]['EXT_SOURCE_1'].dropna(), df[mask_target0]['EXT_SOURCE_1'].dropna()]
data_source2 = [df[mask_target1]['EXT_SOURCE_2'].dropna(), df[mask_target0]['EXT_SOURCE_2'].dropna()]
data_source3 = [df[mask_target1]['EXT_SOURCE_3'].dropna(), df[mask_target0]['EXT_SOURCE_3'].dropna()]
group_labels = ['Default', 'No Default']
colors = ['#333F44', '#37AA9C']


# Figure Source 1
def plot_ext_1():
    fig1 = ff.create_distplot(data_source1, group_labels,
                              show_hist=False,
                              colors=colors,
                              show_rug=False)

    fig1.add_trace(
        go.Scatter(x=[np.array(customer_df_raw['EXT_SOURCE_1'])[0], np.array(customer_df_raw['EXT_SOURCE_1'])[0]],
                   y=[-0.5, 2.5], mode="lines", name='Client', line=go.scatter.Line(color="red")))

    fig1.update_layout(
        title={'text': "Source Extérieure 1", 'xanchor': 'center', 'yanchor': 'top', 'y': 0.9, 'x': 0.5},
        width=900, height=450,
        xaxis_title="Source Ext 1",
        yaxis_title=" ",
        font=dict(size=15, color="#7f7f7f"))

    fig1.update_yaxes(range=[-0.25, 2.4])

    return fig1


st.plotly_chart(plot_ext_1())


# Figure Source 2
def plot_ext_2():
    fig2 = ff.create_distplot(data_source2, group_labels,
                              show_hist=False,
                              colors=colors,
                              show_rug=False)

    fig2.add_trace(
        go.Scatter(x=[np.array(customer_df_raw['EXT_SOURCE_2'])[0], np.array(customer_df_raw['EXT_SOURCE_2'])[0]],
                   y=[-1, 3.5], mode="lines", name='Client',
                   line=go.scatter.Line(color="red")))

    fig2.update_layout(
        title={'text': "Source Extérieure 2", 'xanchor': 'center', 'yanchor': 'top', 'y': 0.9, 'x': 0.5},
        width=900, height=450,
        xaxis_title="Source Ext 2",
        yaxis_title=" ",
        font=dict(size=15, color="#7f7f7f"))

    fig2.update_yaxes(range=[-0.1, 3.1])
    return fig2


st.plotly_chart(plot_ext_2())


# Figure Source 3
def plot_ext_3():
    fig3 = ff.create_distplot(data_source3, group_labels,
                              show_hist=False,
                              colors=colors,
                              show_rug=False)

    fig3.add_trace(
        go.Scatter(x=[np.array(customer_df_raw['EXT_SOURCE_3'])[0], np.array(customer_df_raw['EXT_SOURCE_3'])[0]],
                   y=[-0.5, 3.5], mode="lines", name='Client',
                   line=go.scatter.Line(color="red")))

    fig3.update_layout(
        title={'text': "Source Extérieure 3", 'xanchor': 'center', 'yanchor': 'top', 'y': 0.9, 'x': 0.5},
        width=900, height=450,
        xaxis_title="Source Ext 3",
        yaxis_title=" ",
        font=dict(size=15, color="#7f7f7f"))

    fig3.update_xaxes(range=[0, 0.9])
    fig3.update_yaxes(range=[-0.1, 2.9])

    return fig3


st.plotly_chart(plot_ext_3())