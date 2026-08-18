"""Shared CSS and small style constants for the dashboard."""

from __future__ import annotations

import streamlit as st

RISK_COLORS = {
    "LOW": "#16A34A",
    "MEDIUM": "#D97706",
    "HIGH": "#EA580C",
    "CRITICAL": "#DC2626",
}

CLASS_COLORS = {"NORMAL": "#16A34A", "FAILURE": "#DC2626"}

CHART_COLORS = {
    "primary": "#2563EB",
    "positive": "#DC2626",  # increases failure risk
    "negative": "#2563EB",  # decreases failure risk
    "neutral": "#64748B",
}

_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

.app-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    padding-bottom: 0.9rem;
    margin-bottom: 1.6rem;
}
.app-header .subtitle {
    color: #64748B;
    font-size: 0.95rem;
}

.metric-card {
    background: var(--secondary-background-color, #F1F5F9);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}
.metric-card .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.65rem;
    font-weight: 700;
    color: inherit;
    line-height: 1.15;
}
.metric-card .sublabel {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.25rem;
}

.section-card {
    background: rgba(148, 163, 184, 0.06);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
}

.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
}

.pred-class-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
}

.disclaimer-box {
    background: rgba(217, 119, 6, 0.08);
    border: 1px solid rgba(217, 119, 6, 0.35);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.88rem;
    color: #92400E;
}

.small-muted {
    color: #64748B;
    font-size: 0.85rem;
}

[data-testid="stMetricValue"] {
    font-weight: 700;
}

section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, sublabel: str | None = None) -> str:
    sub_html = f'<div class="sublabel">{sublabel}</div>' if sublabel else ""
    return (
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def risk_badge_html(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#64748B")
    return (
        f'<span class="risk-badge" style="background:{color}1A;color:{color};'
        f'border:1px solid {color}55;">{risk_level} RISK</span>'
    )


def class_badge_html(predicted_class: str) -> str:
    color = CLASS_COLORS.get(predicted_class, "#64748B")
    return (
        f'<span class="pred-class-badge" style="background:{color}1A;color:{color};'
        f'border:1px solid {color}55;">{predicted_class}</span>'
    )
