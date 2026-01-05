"""
Gradio goal programming configuration (parallel to Streamlit).
"""

from typing import Any, Dict, List, Tuple

import gradio as gr


def _objective_catalog() -> List[Tuple[str, str, str]]:
    return [
        ("stress", "Stress végétal (ne pas dépasser)", "Plant stress (do not exceed)"),
        ("drainage", "Drainage (ne pas dépasser)", "Drainage (do not exceed)"),
        ("irrigation", "Irrigation saisonnière (ne pas dépasser)", "Seasonal irrigation (do not exceed)"),
        ("events", "Nombre d'événements d'irrigation (ne pas dépasser)", "Number of irrigation events (do not exceed)"),
    ]


def _default_goal_spec() -> Dict[str, Any]:
    return {
        "targets": {
            "stress_max": 55.0,
            "irrig_max": 250.0,
            "drain_max": 60.0,
            "events_max": 20,
        },
        "priorities": {"P1": ["stress"], "P2": ["drainage"], "P3": ["irrigation"]},
        "tradeoff_style": "Conservateur",
        "lambdas": {"P1": 1000.0, "P2": 10.0, "P3": 0.1},
    }


def _label(text_fr: str, text_en: str, language: str) -> str:
    return text_fr if language == "fr" else text_en


def _compute_lambdas(style: str) -> Dict[str, float]:
    if style.startswith("Conservateur") or style.startswith("Conservative"):
        return {"P1": 1000.0, "P2": 10.0, "P3": 0.1}
    if style.startswith("Équilibré") or style.startswith("Balanced"):
        return {"P1": 300.0, "P2": 3.0, "P3": 0.03}
    return {"P1": 100.0, "P2": 2.0, "P3": 0.2}


def build_goal_programming_config(language: str = "fr") -> Dict[str, gr.Component]:
    labels = {
        "header": "🎯 Programmation par objectifs lexicographique / Lexicographic goal programming",
        "note": "Définissez les objectifs saisonniers et leurs priorités (lexicographique). / Set seasonal goals and lexicographic priorities.",
        "enable": "Activer les objectifs saisonniers et les priorités / Enable seasonal objectives and priorities",
        "targets": "Objectifs saisonniers / Seasonal targets",
        "stress": "Stress maximal (kPa) / Max plant stress (kPa)",
        "irrig": "Irrigation saisonnière max (mm) / Max seasonal irrigation (mm)",
        "drain": "Drainage saisonnier max (mm) / Max seasonal drainage (mm)",
        "events": "Nombre max d'événements / Max irrigation events",
        "priorities": "Priorités (tiers lexicographiques) / Priorities (lexicographic tiers)",
        "p1": "Priorité P1 (plus important) / Priority P1 (highest)",
        "p2": "Priorité P2 / Priority P2",
        "p3": "Priorité P3 (moins important) / Priority P3 (lowest)",
        "tradeoff": "Style de compromis / Trade-off style",
        "apply": "Appliquer les objectifs / Apply objectives",
    }

    defaults = _default_goal_spec()
    options = _objective_catalog()
    option_ids = [o[0] for o in options]
    option_labels = {o[0]: (o[1] if language == "fr" else o[2]) for o in options}

    gr.Markdown(f"### {labels['header']}")
    gr.Markdown(labels["note"])
    enable = gr.Checkbox(label=labels["enable"], value=False)

    gr.Markdown(f"**{labels['targets']}**")
    with gr.Row():
        stress_max = gr.Slider(0.0, 200.0, value=float(defaults["targets"]["stress_max"]), step=1.0, label=labels["stress"])
        irrig_max = gr.Slider(0.0, 600.0, value=float(defaults["targets"]["irrig_max"]), step=5.0, label=labels["irrig"])
    with gr.Row():
        drain_max = gr.Slider(0.0, 300.0, value=float(defaults["targets"]["drain_max"]), step=5.0, label=labels["drain"])
        events_max = gr.Slider(0, 60, value=int(defaults["targets"]["events_max"]), step=1, label=labels["events"])

    gr.Markdown(f"**{labels['priorities']}**")
    with gr.Row():
        P1 = gr.Dropdown(option_ids, value=defaults["priorities"]["P1"], label=labels["p1"], multiselect=True)
        P2 = gr.Dropdown(option_ids, value=defaults["priorities"]["P2"], label=labels["p2"], multiselect=True)
        P3 = gr.Dropdown(option_ids, value=defaults["priorities"]["P3"], label=labels["p3"], multiselect=True)

    styles = [
        "Conservateur (satisfaction d'abord) / Conservative (prioritize goal satisfaction)",
        "Équilibré / Balanced",
        "Agressif (plus d'échanges) / Aggressive (allow trade-offs)",
    ]
    style = gr.Radio(styles, value=styles[0], label=labels["tradeoff"])
    apply_btn = gr.Button(labels["apply"])

    return {
        "enable": enable,
        "stress_max": stress_max,
        "irrig_max": irrig_max,
        "drain_max": drain_max,
        "events_max": events_max,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "style": style,
        "apply": apply_btn,
        "option_labels": option_labels,
    }


def build_goal_spec(
    enable: bool,
    stress_max: float,
    irrig_max: float,
    drain_max: float,
    events_max: float,
    P1: List[str],
    P2: List[str],
    P3: List[str],
    style: str,
) -> Dict[str, Any]:
    if not enable:
        return {}
    return {
        "targets": {
            "stress_max": float(stress_max),
            "irrig_max": float(irrig_max),
            "drain_max": float(drain_max),
            "events_max": int(events_max),
        },
        "priorities": {"P1": P1 or [], "P2": P2 or [], "P3": P3 or []},
        "tradeoff_style": style,
        "lambdas": _compute_lambdas(style),
    }
