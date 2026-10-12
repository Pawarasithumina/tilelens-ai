"""Reusable visual styling for the TileLens AI Streamlit dashboard."""

import streamlit as st

CSS = r"""
<style>
/* Main shell */
.block-container {max-width: 1540px !important; padding-top: 2.0rem !important; padding-bottom: 3.8rem !important;}
section[data-testid="stSidebar"] {border-right: 1px solid #253248;}
section[data-testid="stSidebar"] .block-container {padding-top: 1.3rem !important;}
h1, h2, h3 {letter-spacing: -0.035em !important;}
h1 {font-size: 2.25rem !important; font-weight: 740 !important;}
h2 {font-size: 1.5rem !important;}
h3 {font-size: 1.15rem !important;}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #18263B 0%, #121D30 100%);
    border: 1px solid #2A3C54;
    border-radius: 15px;
    padding: 18px 19px;
    min-height: 124px;
}
[data-testid="stMetricLabel"] {color: #9EB2CA; font-size: .92rem;}
[data-testid="stMetricValue"] {font-weight: 720; letter-spacing: -0.03em;}
[data-testid="stFileUploader"] {border-radius: 14px;}
[data-testid="stFileUploaderDropzone"] {border: 1px dashed #47637D; border-radius: 14px; background: #121F32;}
[data-testid="stDataFrame"] {border: 1px solid #263951; border-radius: 12px; overflow: hidden;}
[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stExpander"] {border: 1px solid #2B3D54; border-radius: 12px;}
.stButton button[kind="primary"] {border-radius: 10px; font-weight: 650;}
.stDownloadButton button {border-radius: 10px;}
[data-testid="stSidebar"] [role="radiogroup"] label {border-radius: 9px; padding: 0.48rem 0.5rem;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: #202D43;}
/* Branded text and hero */
.tl-brand {display:flex; align-items:center; gap:12px; padding: 6px 0 8px 0;}
.tl-brand-symbol {width:40px; height:40px; display:grid; place-items:center; background:#173E47; border:1px solid #2D857E; color:#66E0CE; border-radius:12px; font-size:1.8rem;}
.tl-brand-name {font-weight:780; font-size:1.38rem; letter-spacing:-.045em;}
.tl-brand-name span {color:#63E4D2;}
.tl-brand-tag {font-size:.67rem; color:#8A9CB6; font-weight:700; letter-spacing:.16em;}
.tl-sidebar-label {font-size:.72rem; color:#8DA5C2; letter-spacing:.13em; font-weight:750; margin-bottom:.5rem;}
.tl-eyebrow {font-size:.73rem; color:#58DBC9; font-weight:760; letter-spacing:.145em; padding-top:4px;}
.tl-title {font-size:2.1rem; font-weight:780; letter-spacing:-.045em; line-height:1.15; margin:0.28rem 0 0.45rem;}
.tl-subtitle {font-size:1rem; color:#9FAFC5; margin-bottom:1.1rem;}
.tl-hero {background:linear-gradient(115deg,#182B43,#102333 65%,#153E40); border:1px solid #305366; border-radius:18px; padding:22px 26px; margin:1rem 0 1.6rem;}
.tl-hero strong {font-size:1.16rem; font-weight:760;}
.tl-hero p {margin:.4rem 0 0; color:#AEC2D6; line-height:1.55;}
.tl-footer {border-top:1px solid #27354A; color:#8498AF; font-size:.77rem; padding-top:20px; margin-top:45px;}
.tl-kicker {font-size:.78rem; letter-spacing:.09em; color:#78E1D2; font-weight:750; margin:18px 0 10px;}
.tl-empty {border:1px dashed #36516D; padding:26px; border-radius:14px; background:#141F30; color:#ABBCD0;}
</style>
"""


def apply_dashboard_styles():
    st.markdown(CSS, unsafe_allow_html=True)


def page_heading(title, subtitle, eyebrow):
    # Only fixed code-authored strings may be interpolated into this HTML.
    st.markdown(
        f'<div class="tl-eyebrow">{eyebrow}</div>'
        f'<div class="tl-title">{title}</div>'
        f'<div class="tl-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )
