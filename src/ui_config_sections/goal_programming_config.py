"""
Bloc UI de goal programming (Option A) avec objectifs ciblés et priorités.
"""
from typing import Any, Dict, List, Tuple

import streamlit as st


def _objective_catalog() -> List[Tuple[str, str, str]]:
    """
    Référentiel des objectifs :
    (id, libellé FR, libellé EN)
    """
    return [
        ("stress", "Stress végétal (ne pas dépasser)", "Plant stress (do not exceed)"),
        ("drainage", "Drainage (ne pas dépasser)", "Drainage (do not exceed)"),
        ("irrigation", "Irrigation saisonnière (ne pas dépasser)", "Seasonal irrigation (do not exceed)"),
        ("events", "Nombre d'événements d'irrigation (ne pas dépasser)", "Number of irrigation events (do not exceed)"),
    ]


def _default_goal_spec() -> Dict[str, Any]:
    """Spécification par défaut des objectifs (ciblage + priorités)."""
    return {
        "targets": {
            "stress_max": 55.0,
            "irrig_max": 250.0,
            "drain_max": 60.0,
            "events_max": 20,
        },
        "priorities": {
            "P1": ["stress"],
            "P2": ["drainage"],
            "P3": ["irrigation"],
        },
        "tradeoff_style": "Conservateur",
        "lambdas": {"P1": 1000.0, "P2": 10.0, "P3": 0.1},
    }


def _label(text_fr: str, text_en: str, language: str) -> str:
    """Retourne le libellé dans la langue choisie."""
    return text_fr if language == "fr" else text_en


def _compute_lambdas(style: str) -> Dict[str, float]:
    """Calcule les lambdas en fonction du style de compromis."""
    if style.startswith("Conservateur") or style.startswith("Conservative"):
        return {"P1": 1000.0, "P2": 10.0, "P3": 0.1}
    if style.startswith("Équilibré") or style.startswith("Balanced"):
        return {"P1": 300.0, "P2": 3.0, "P3": 0.03}
    return {"P1": 100.0, "P2": 2.0, "P3": 0.2}


def _objective_choices(language: str) -> List[Tuple[str, str]]:
    """Liste (id, libellé affiché) pour les tiers lexicographiques (mono-langue)."""
    catalog = _objective_catalog()
    return [(obj_id, en if language == "en" else fr) for obj_id, fr, en in catalog]


def _id_to_label(obj_id: str, language: str) -> str:
    """Retourne le libellé dans la langue courante pour un id d'objectif."""
    for oid, fr, en in _objective_catalog():
        if oid == obj_id:
            return en if language == "en" else fr
    return obj_id


def _normalize_priorities(values: List[str]) -> List[str]:
    """
    Convertit d'anciens libellés (FR/EN) en ids canoniques.
    Utile pour rétrocompatibilité avec les sélections existantes.
    """
    label_to_id = {}
    for oid, fr, en in _objective_catalog():
        label_to_id[fr] = oid
        label_to_id[en] = oid
        label_to_id[f"{fr} / {en}"] = oid
    normalized = []
    for v in values:
        normalized.append(label_to_id.get(v, v))
    return normalized


def render_goal_programming_config(language: str = "fr") -> Dict[str, Any]:
    """
    Rend le bloc Goal Programming (Option A) dans la sidebar avec :
    - Objectifs ciblés (stress, irrigation, drainage, événements)
    - Priorités lexicographiques (P1 ≻ P2 ≻ P3)
    - Style de compromis → échelles λ
    - Section robustesse météo (identique à la version précédente)
    """
    labels = {
        "header": _label("🎯 Programmation par objectifs lexicographique", "🎯 Lexicographic goal programming", language),
        "note": _label(
            "Définissez les objectifs saisonniers et leurs priorités (lexicographique).",
            "Set seasonal goals and lexicographic priorities.",
            language,
        ),
        "enable": _label("Activer les objectifs saisonniers et les priorités", "Enable seasonal objectives and priorities", language),
        "targets": _label("Objectifs saisonniers", "Seasonal targets", language),
        "stress": _label("Stress maximal (kPa)", "Max plant stress (kPa)", language),
        "irrig": _label("Irrigation saisonnière max (mm)", "Max seasonal irrigation (mm)", language),
        "drain": _label("Drainage saisonnier max (mm)", "Max seasonal drainage (mm)", language),
        "events": _label("Nombre max d'événements", "Max irrigation events", language),
        "priorities": _label("Priorités (tiers lexicographiques)", "Priorities (lexicographic tiers)", language),
        "p1": _label("Priorité P1 (plus important)", "Priority P1 (highest)", language),
        "p2": _label("Priorité P2", "Priority P2", language),
        "p3": _label("Priorité P3 (moins important)", "Priority P3 (lowest)", language),
        "tradeoff": _label("Style de compromis", "Trade-off style", language),
        "apply": _label("Appliquer les objectifs", "Apply objectives", language),
        "applied": _label("Objectifs appliqués.", "Objectives applied.", language),
        "duplicate": _label(
            "Objectifs dupliqués entre priorités",
            "Duplicate objectives across tiers",
            language,
        ),
        "summary_targets": _label("Cibles actuelles", "Current targets", language),
        "summary_priorities": _label("Priorités", "Priorities", language),
        "summary_lambdas": _label("Échelles (λ)", "Penalty scales (λ)", language),
        "robust": _label("Évaluer robustesse (déplacement météo)", "Evaluate robustness (weather shift)", language),
        "rain_shift": _label("Décalage pluie (%)", "Rain shift (%)", language),
        "temp_shift": _label("Décalage ET0 (%)", "ET0 shift (%)", language),
        "shift_mode": _label("Mode d'application des décalages météo", "Weather shift mode", language),
        "shocks_days": _label("Nombre de jours de choc", "Number of shock days", language),
        "shocks_window": _label("Taille de fenêtre max pour les chocs", "Max window size for shocks", language),
        "shift_modes": {
            "none": _label("Aucun (pas de décalage)", "None (no shift)", language),
            "seasonal": _label("Biais saison entier", "Season-wide bias", language),
            "shocks": _label("Chocs aléatoires (quelques jours)", "Random shock days", language),
        },
    }

    # Initialisation des états
    if "goal_spec" not in st.session_state:
        st.session_state["goal_spec"] = _default_goal_spec()
    if "proposal_a_config" not in st.session_state:
        st.session_state["proposal_a_config"] = {}

    st.markdown(f'<h3 class="section-header">{labels["header"]}</h3>', unsafe_allow_html=True)
    st.caption(labels["note"])

    enable = st.checkbox(labels["enable"], value=st.session_state["proposal_a_config"].get("enabled", False), key="proposal_a_enable")

    # Cibles saisonnières
    st.markdown(f"**{labels['targets']}**")
    targets = st.session_state["goal_spec"]["targets"]
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        stress_max = st.slider(labels["stress"], 0.0, 200.0, float(targets["stress_max"]), 1.0)
        drain_max = st.slider(labels["drain"], 0.0, 300.0, float(targets["drain_max"]), 5.0)
    with col_t2:
        irrig_max = st.slider(labels["irrig"], 0.0, 600.0, float(targets["irrig_max"]), 5.0)
        events_max = st.slider(labels["events"], 0, 60, int(targets["events_max"]), 1)

    # Priorités lexicographiques
    st.markdown(f"**{labels['priorities']}**")
    choices = _objective_choices(language)
    cur_priorities = st.session_state["goal_spec"]["priorities"]

    # Rétrocompatibilité : convertir les anciens libellés en ids
    cur_priorities = {
        k: _normalize_priorities(v) for k, v in cur_priorities.items()
    }

    def _safe_defaults(key: str) -> List[str]:
        return [obj for obj in cur_priorities.get(key, []) if obj in [c[0] for c in choices]]

    def _fmt(choice_id: str) -> str:
        return _id_to_label(choice_id, language)

    option_ids = [c[0] for c in choices]
    P1 = st.multiselect(labels["p1"], options=option_ids, format_func=_fmt, default=_safe_defaults("P1"))
    P2 = st.multiselect(labels["p2"], options=option_ids, format_func=_fmt, default=_safe_defaults("P2"))
    P3 = st.multiselect(labels["p3"], options=option_ids, format_func=_fmt, default=_safe_defaults("P3"))

    # Style de compromis
    styles = [
        _label("Conservateur (satisfaction d'abord)", "Conservative (prioritize goal satisfaction)", language),
        _label("Équilibré", "Balanced", language),
        _label("Agressif (plus d'échanges)", "Aggressive (allow trade-offs)", language),
    ]
    cur_style = st.session_state["goal_spec"].get("tradeoff_style", styles[0])
    style = st.radio(labels["tradeoff"], styles, index=styles.index(cur_style) if cur_style in styles else 0)
    lambdas = _compute_lambdas(style)

    # Validation des duplications entre tiers
    duplicates = (set(P1) & set(P2)) | (set(P1) & set(P3)) | (set(P2) & set(P3))
    if duplicates:
        st.warning(f"{labels['duplicate']}: {', '.join(_id_to_label(x, language) for x in sorted(duplicates))}")

    if st.button(labels["apply"], use_container_width=True):
        st.session_state["goal_spec"] = {
            "targets": {
                "stress_max": float(stress_max),
                "irrig_max": float(irrig_max),
                "drain_max": float(drain_max),
                "events_max": int(events_max),
            },
            "priorities": {"P1": P1, "P2": P2, "P3": P3},
            "tradeoff_style": style,
            "lambdas": lambdas,
        }
        st.success(labels["applied"])

    # Résumé rapide (sidebar)
    spec = st.session_state["goal_spec"]
    st.markdown(f"**{labels['summary_targets']}**")
    st.write(
        {
            labels["stress"]: spec["targets"]["stress_max"],
            labels["irrig"]: spec["targets"]["irrig_max"],
            labels["drain"]: spec["targets"]["drain_max"],
            labels["events"]: spec["targets"]["events_max"],
        }
    )
    st.markdown(f"**{labels['summary_priorities']}**")
    st.write({k: [_id_to_label(v, language) for v in vals] for k, vals in spec["priorities"].items()})
    st.markdown(f"**{labels['summary_lambdas']}**")
    st.write(spec["lambdas"])

    st.markdown("---")
    # Section robustesse météo (héritée de la version précédente)
    robust = st.checkbox(labels["robust"], value=st.session_state["proposal_a_config"].get("robustness_eval", False), key="proposal_a_robust")
    colr1, colr2 = st.columns(2)
    with colr1:
        rain_shift = st.slider(labels["rain_shift"], -50, 50, int(st.session_state["proposal_a_config"].get("rain_shift_pct", 0)), 5)
    with colr2:
        temp_shift = st.slider(labels["temp_shift"], -30, 30, int(st.session_state["proposal_a_config"].get("et0_shift_pct", 0)), 5)

    shift_options = list(labels["shift_modes"].keys())
    default_shift = st.session_state["proposal_a_config"].get("shift_mode", "none")
    try:
        default_idx = shift_options.index(default_shift)
    except ValueError:
        default_idx = 0
    shift_mode = st.selectbox(
        labels["shift_mode"],
        options=shift_options,
        format_func=lambda k: labels["shift_modes"][k],
        index=default_idx,
        key="proposal_a_shift_mode",
    )
    shock_days = st.number_input(
        labels["shocks_days"],
        min_value=0,
        max_value=30,
        value=int(st.session_state["proposal_a_config"].get("shock_days", 0)),
        step=1,
        key="proposal_a_shock_days",
        help=_label("Appliqué si chocs aléatoires", "Used when random shocks are selected", language),
    )
    shock_window = st.number_input(
        labels["shocks_window"],
        min_value=1,
        max_value=30,
        value=int(st.session_state["proposal_a_config"].get("shock_window", 7)),
        step=1,
        key="proposal_a_shock_window",
        help=_label("Appliqué si chocs aléatoires", "Used when random shocks are selected", language),
    )

    config = {
        "enabled": enable,
        "goal_spec": st.session_state["goal_spec"],
        "lambdas": [lambdas["P1"], lambdas["P2"], lambdas["P3"]],
        "robustness_eval": robust,
        "rain_shift_pct": rain_shift,
        "et0_shift_pct": temp_shift,
        "shift_mode": shift_mode,
        "shock_days": shock_days,
        "shock_window": shock_window,
        # Champs hérités (compatibilité arrière)
        "teacher_weight": 0.0,
        "residual_alpha": 0.0,
        "stability_margin": 0.0,
        "feasibility_penalty": 0.0,
    }
    st.session_state.proposal_a_config = config
    return config
