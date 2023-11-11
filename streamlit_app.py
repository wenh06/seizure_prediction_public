import json
from copy import deepcopy

import pandas as pd
import streamlit as st

from config import ServingConfig
from models import SeizurePredictionModel


st.set_page_config(
    page_title="Seizure Prediction",
    page_icon="📋",
    layout="centered",
)


@st.cache_resource
def get_model():
    serving_config = deepcopy(ServingConfig)
    loaded_model = SeizurePredictionModel.from_file(serving_config.model_path)
    return loaded_model


model = get_model()


st.title("Seizure Prediction")

tab_compute, tab_upload = st.tabs(["Direct Compute", "Compute from File"])

with tab_compute:
    data = st.text_area(
        label="Input of JSON format",
        value="",
        height=200,
    )

    compute_button = st.button(label="Compute")

    if compute_button:
        try:
            data = json.loads(data)
        except Exception as e:
            st.error(e)
            st.stop()
        if (
            (not isinstance(data, (dict, list)))
            or (len(data) == 0)
            or (isinstance(data, list) and any([not isinstance(d, dict) for d in data]))
        ):
            st.error("Input data must be a non-empty dict or a non-empty list of dict")
            st.stop()

        try:
            result = model.pipeline(data)
            st.json(result)
        except Exception as e:
            st.error(e)
            st.stop()


with tab_upload:
    uploaded_file = st.file_uploader(
        label="Upload a file (CSV or Excel or JSON)",
        type=["csv", "xlsx", "json"],
    )

    if uploaded_file is None:
        st.error("Please upload a file.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            data = uploaded_file.read()
            data = json.loads(data)
    except Exception as e:
        st.error(e)
        st.stop()

    try:
        result = model.pipeline(data)
        st.json(result)
    except Exception as e:
        st.error(e)
        st.stop()
