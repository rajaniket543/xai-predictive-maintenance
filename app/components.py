"""Reusable UI building blocks shared across dashboard pages."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from styles import CHART_COLORS, class_badge_html, risk_badge_html

PLOTLY_TEMPLATE = "plotly_white"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="app-header"><div><h1 style="margin-bottom:0">{title}</h1></div></div>'
        f'<p class="subtitle" style="margin-top:-1rem;color:#64748B;">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#0F172A", "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 30], "color": "#DCFCE7"},
                    {"range": [30, 60], "color": "#FEF9C3"},
                    {"range": [60, 80], "color": "#FED7AA"},
                    {"range": [80, 100], "color": "#FECACA"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
            title={"text": "Failure Probability", "font": {"size": 16}},
        )
    )
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def render_prediction_result(result: dict[str, Any]) -> None:
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.plotly_chart(
            probability_gauge(result["failure_probability"], result["threshold_used"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col2:
        st.markdown(
            f'<div class="section-card">'
            f'<div class="small-muted">PREDICTED CLASS</div>'
            f'<div style="margin:0.35rem 0 0.9rem 0">{class_badge_html(result["predicted_class"])}</div>'
            f'<div class="small-muted">RISK LEVEL</div>'
            f'<div style="margin:0.35rem 0 0.9rem 0">{risk_badge_html(result["risk_level"])}</div>'
            f'<div class="small-muted">RECOMMENDED ACTION</div>'
            f'<div style="margin-top:0.35rem;font-weight:600">{result["risk_action"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def contributor_bar_chart(
    features: list[str], values: list[float], title: str, x_label: str = "Contribution"
) -> go.Figure:
    order = sorted(range(len(values)), key=lambda i: values[i])
    features_sorted = [features[i] for i in order]
    values_sorted = [values[i] for i in order]
    colors = [CHART_COLORS["positive"] if v > 0 else CHART_COLORS["negative"] for v in values_sorted]

    fig = go.Figure(
        go.Bar(
            x=values_sorted,
            y=features_sorted,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#94A3B8")
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        template=PLOTLY_TEMPLATE,
        height=max(320, 28 * len(features_sorted)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="disclaimer-box">{text}</div>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.info(message)
