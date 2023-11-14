import json
from copy import deepcopy  # noqa: F401
from typing import List

import pandas as pd
import requests
import streamlit as st

from config import DataPreprocessConfig, ServingConfig  # noqa: F401
from models import SeizurePredictionModel  # noqa: F401

st.set_page_config(
    page_title="Seizure Prediction",
    page_icon=":hospital:",
    layout="centered",
)


FLASK_SERVING_URL = "http://43.140.244.112:11111/seizure_prediction"


def fetch_prediction(data: List[dict]) -> dict:
    try:
        response = requests.post(
            url=FLASK_SERVING_URL,
            json=data,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        result = {
            "code": 4,
            "error_type": type(e).__name__,
            "error_msg": str(e),
        }
    return result


# @st.cache_resource
# def get_model():
#     serving_config = deepcopy(ServingConfig)
#     loaded_model = SeizurePredictionModel.from_file(serving_config.model_path)
#     return loaded_model


# model = get_model()


st.title("Seizure Prediction")

st.sidebar.title("Configuration")

language = st.sidebar.selectbox(
    label="Language (语言)",
    options=["English", "中文"],
    key="language",
)
output_prob = st.sidebar.toggle(
    label="Output Confidence" if language == "English" else "输出置信度",
    value=False,
    key="output_prob",
)
output_format = st.sidebar.selectbox(
    label="Output Format" if language == "English" else "输出格式",
    options=["JSON", "plain text" if language == "English" else "文本"],
    key="output_format",
)


dbci_app_url = "https://diff-binom-confint.streamlit.app/"

for _ in range(5):
    st.sidebar.write("\n")
st.sidebar.markdown("**For the computation of binomial confidence intervals, please visit:**")
st.sidebar.markdown(
    f'<p style="text-align: center;"><a href="{dbci_app_url}" target="_blank">Diff Binom CI</a></p>',
    unsafe_allow_html=True,
)


def process_output(output_results: List[dict]):
    st.markdown("**Results**")
    if output_format == "JSON":
        refined_results = []
        for item in output_results:
            # keys in item: "prediction", "probability"
            refined_result = {}
            if item["prediction"] == 1:
                refined_result["prediction"] = "Positive"
            else:
                refined_result["prediction"] = "Negative"
            if output_prob:
                refined_result["confidence"] = item["probability"]
            refined_results.append(refined_result)
        st.json(refined_results)
    else:  # plain text
        refined_results = ""
        for idx, item in enumerate(output_results):
            if idx > 0:
                refined_results += "\n"
            refined_results += f"Case {idx + 1}: "
            if item["prediction"] == 1:
                refined_results += "Positive"
            else:
                refined_results += "Negative"
            if output_prob:
                refined_results += f", confidence: {item['probability']}"
        st.code(refined_results)


tab_direct, tab_json, tab_upload = st.tabs(["Direct Computation", "Compute from JSON", "Compute from File"])

with tab_direct:
    gender = st.selectbox(
        label="Gender" if language == "English" else "性别",
        options=["Male", "Female"] if language == "English" else ["男", "女"],
        key="gender",
    )
    age = st.number_input(
        label="Age" if language == "English" else "年龄",
        min_value=0,
        max_value=200,
        value=20,
        step=1,
        key="age",
    )
    extent_of_resection = st.selectbox(
        label="Extent of Resection" if language == "English" else "手术切除方式",
        options=["Gross Total", "Subtotal", "Partial"] if language == "English" else ["部分切除", "大部切除", "次全切", "近全切", "全切"],
        key="extent_of_resection",
    )
    grading_WHO = st.selectbox(
        label="WHO Grading" if language == "English" else "病理分级",
        options=["WHO I", "WHO II", "WHO III", "WHO IV"],
        key="grading_WHO",
    )
    # fmt: off
    region_options = {
        "English": [
            "Frontal", "Parietal", "Temporal", "Occipital", "Insular", "Ventricle", "Cerebellum", "Thalamus", "Corpus Callosum",
            "Basal Ganglia", "Sellar Region", "Brainstem", "Others"
        ],
        "中文": [
            "额", "顶", "颞", "枕", "岛", "脑室", "小脑", "丘脑", "胼胝体", "基底节", "鞍区", "脑干", "其他"
        ],
    }
    # fmt: on
    region_involved = st.selectbox(
        label="Region Involved" if language == "English" else "肿瘤分区",
        options=region_options[language],
        key="region_involved",
    )
    # fmt: off
    pathology_options = {
        "English": [
            "Glioblastoma Multiforme", "Anaplastic Astrocytoma", "Oligoastrocytoma", "Asrocytoma", "PilocyticAstrocytoma",
            "Neurocytoma", "Mixed", "Others"
        ],
        "中文": [
            "分型胶质母", "分型间变型星形", "分型少突星形", "分型星形", "分型毛细胞星形", "分型中枢神经", "分型混合", "分型其他",
        ],
    }
    # fmt: on
    pathology = st.selectbox(
        label="Pathology" if language == "English" else "病理分型",
        options=pathology_options[language],
        key="pathology",
    )
    # fmt: off
    comorbidity_options = {
        "English": [
            "Hyponatremia", "Hypoproteinemia", "Hypokalemia", "Hyperchloremia", "Hypochloremia",
            "Central Nervous System Infection", "Hydrocephalus", "Coagulation Disorders",
        ],
        "中文": [
            "低钠血症", "低蛋白血症", "低钾血症", "高氯血症", "低氯血症", "中枢神经感染", "脑积水", "凝血功能异常",
        ],
    }
    # fmt: on
    comorbidity = st.multiselect(
        label="Comorbidity" if language == "English" else "合并症",
        options=comorbidity_options[language],
        key="comorbidity",
    )
    surgery_duration = st.number_input(
        label="Surgery Duration (hours)" if language == "English" else "手术时长（小时）",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=0.1,
        key="surgery_duration",
    )
    maximum_diameter = st.number_input(
        label="Maximum Diameter (cm)" if language == "English" else "肿瘤大小（cm）",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=0.1,
        key="maximum_diameter",
    )
    bleeding_amount = st.number_input(
        label="Bleeding Amount (ml)" if language == "English" else "出血量（ml）",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=0.1,
        key="bleeding_amount",
    )

    Ki_67 = st.number_input(
        label="Ki-67",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=0.1,
        key="Ki_67",
    )
    IDH1_R132 = st.selectbox(
        label="IDH1 R132",
        options=["+++", "++", "+", "±", "-"],
        key="IDH1_R132",
    )

    decompressive_craniectomy = st.checkbox(
        label="Decompressive Craniectomy" if language == "English" else "去骨瓣减压术",
        value=False,
        key="decompressive_craniectomy",
    )
    complication_infection = st.checkbox(
        label="Complication Infection" if language == "English" else "并发症感染",
        value=False,
        key="complication_infection",
    )
    recurrent_glioma = st.checkbox(
        label="Recurrent Glioma" if language == "English" else "复发胶质瘤",
        value=False,
        key="recurrent_glioma",
    )

    compute_button = st.button(label="Predict", key="compute_direct")

    if compute_button:
        if maximum_diameter <= 0:
            if language == "English":
                st.error("Maximum Diameter must be greater than 0.")
            else:
                st.error("肿瘤大小必须大于0")
            st.stop()
        pathology = pathology.replace(" ", "")
        extent_of_resection = extent_of_resection.replace(" ", "")
        en2zh_mapping = {v: k for k, v in DataPreprocessConfig.zh2en_mapping.items()}
        compute_input = {
            "性别": en2zh_mapping.get(gender, gender),
            "年龄": age,
            "手术时长": surgery_duration if surgery_duration > 0 else None,
            "肿瘤大小": maximum_diameter,
            "出血量": bleeding_amount if bleeding_amount > 0 else None,
            "并发症感染": 1 if complication_infection else 0,
            "手术切除方式": en2zh_mapping.get(extent_of_resection, extent_of_resection),
            "病理分级": grading_WHO.replace(" ", "") + "级",
            "复发胶质瘤": 1 if recurrent_glioma else 0,
            "去骨瓣减压术": 1 if decompressive_craniectomy else 0,
            "肿瘤分区": en2zh_mapping.get(region_involved, region_involved),
            "病理分型粗": en2zh_mapping.get(pathology, pathology),
            "BIO_Ki-67": Ki_67 if Ki_67 > 0 else None,
            "BIO_IDH1-R132": IDH1_R132,
            # categorized features
            "C肿瘤最大直径": "<5" if maximum_diameter < 5 else ">=5",
        }
        for idx, item in enumerate(comorbidity_options["中文"]):
            if language == "English":
                compute_input[f"合并症_{item}"] = 1 if comorbidity_options["English"][idx] in comorbidity else 0
            else:
                compute_input[f"合并症_{item}"] = 1 if item in comorbidity else 0
        try:
            # result = model.pipeline(compute_input)
            result = fetch_prediction(compute_input)
            if result["code"] != 0:
                st.error(f"Error: {result['error_type']}: {result['error_msg']}")
                st.stop()
            else:
                result = result["result"]
            process_output(result)
        except Exception as e:
            st.error(e)
            st.stop()

with tab_json:
    data = st.text_area(
        label="Input of JSON format",
        value="",
        height=200,
    )

    compute_button = st.button(label="Predict", key="compute_input")

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
            if language == "English":
                st.error("Input data must be a non-empty dict or a non-empty list of dict")
            else:
                st.error("输入数据必须是非空字典或非空字典列表")
            st.stop()

        try:
            # result = model.pipeline(data)
            result = fetch_prediction(data)
            if result["code"] != 0:
                st.error(f"Error: {result['error_type']}: {result['error_msg']}")
                st.stop()
            else:
                result = result["result"]
            process_output(result)
        except Exception as e:
            st.error(e)
            st.stop()


with tab_upload:
    uploaded_file = st.file_uploader(
        label="Upload a file (CSV or Excel or JSON)" if language == "English" else "上传文件（文件格式为CSV或Excel或JSON）",
        type=["csv", "xlsx", "json"],
    )

    compute_button = st.button(label="Predict", key="compute_upload")

    if compute_button:

        if uploaded_file is None:
            if language == "English":
                st.error("Please upload a file.")
            else:
                st.error("请上传文件")
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
            # result = model.pipeline(data)
            result = fetch_prediction(data)
            if result["code"] != 0:
                st.error(f"Error: {result['error_type']}: {result['error_msg']}")
                st.stop()
            else:
                result = result["result"]
            process_output(result)
        except Exception as e:
            st.error(e)
            st.stop()


# run command:
# nohup streamlit run streamlit_app.py  > log/app.log 2>&1 & echo $! > log/app.pid
