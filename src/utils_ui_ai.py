"""
Utilitaires pour l'interface assistant IA dans la sidebar Streamlit.
"""

from typing import Dict, Any

import numpy as np
import streamlit as st

from src.llm_local import LocalLLM, LLMConfig
from src.utils_physics_config import get_rule_seuil_unique_config


def render_ai_assistant_sidebar(current_lang: str) -> None:
    assistant_heading = "🤖 Assistant IA" if current_lang == "fr" else "🤖 AI Assistant"
    with st.expander(assistant_heading, expanded=False):
        st.caption(
            "Interaction locale (Ollama, llama.cpp, Transformers)"
            if current_lang == "fr"
            else "Local interaction (Ollama, llama.cpp, Transformers)"
        )
        llm_backend = st.selectbox(
            "Backend",
            options=["ollama", "llamacpp", "transformers"],
            index=0,
            help=(
                "Ollama est le plus simple pour du local. "
                "llama.cpp est léger. Transformers est flexible mais plus lourd."
                if current_lang == "fr"
                else "Ollama is the easiest local option. "
                     "llama.cpp is lightweight. Transformers is flexible but heavier."
            ),
            key="llm_backend",
        )
        llm_model = st.text_input(
            "Modèle" if current_lang == "fr" else "Model",
            value="llama3.2:1b",
            help=(
                "Nom exact du modèle (voir `ollama list`). Ex: `llama3.2:1b`."
                if current_lang == "fr"
                else "Exact model name (see `ollama list`). Example: `llama3.2:1b`."
            ),
            key="llm_model",
        )
        llm_endpoint = st.text_input(
            "Endpoint (optionnel)" if current_lang == "fr" else "Endpoint (optional)",
            value="http://127.0.0.1:11434",
            key="llm_endpoint",
            help="ex: http://localhost:11434 (Ollama) ou http://localhost:8080 (llama.cpp)"
            if current_lang == "fr"
            else "e.g., http://localhost:11434 (Ollama) or http://localhost:8080 (llama.cpp)",
        )
        llm_temperature = st.slider(
            "Température" if current_lang == "fr" else "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help=(
                "Faible = réponses plus factuelles; élevé = plus créatif. "
                "Pour l'analyse des données, 0.1–0.3 est recommandé."
                if current_lang == "fr"
                else "Low = more factual; high = more creative. "
                     "For data analysis, 0.1–0.3 is recommended."
            ),
            key="llm_temperature",
        )
        llm_max_tokens = st.number_input(
            "Max tokens" if current_lang == "fr" else "Max tokens",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
            help=(
                "Limite la longueur de la réponse. Pour des analyses synthétiques, 256–800 suffit."
                if current_lang == "fr"
                else "Limits response length. For concise analyses, 256–800 is usually enough."
            ),
            key="llm_max_tokens",
        )
        llm_system = st.text_area(
            "Instruction système" if current_lang == "fr" else "System instruction",
            value="Tu es un assistant spécialisé en irrigation intelligente et RL."
            if current_lang == "fr"
            else "You are an assistant specialized in smart irrigation and RL.",
            height=80,
            key="llm_system",
        )
        llm_strict = st.checkbox(
            "Mode strict (pas d'hallucination)"
            if current_lang == "fr"
            else "Strict mode (no hallucination)",
            value=True,
            key="llm_strict",
        )
        llm_prompt_lang = st.selectbox(
            "Langue du prompt" if current_lang == "fr" else "Prompt language",
            options=["auto", "fr", "en"],
            index=0,
            key="llm_prompt_lang",
        )
        llm_translate_prompt = st.checkbox(
            "Traduire le prompt vers la langue de l'UI"
            if current_lang == "fr"
            else "Translate prompt to UI language",
            value=False,
            key="llm_translate_prompt",
        )
        llm_prompt = st.text_area(
            "Votre question" if current_lang == "fr" else "Your question",
            value=(
                "Analyse les données des graphiques (psi, S, irrigation/pluie, ETc, drainage). "
                "Donne: % de jours dans la bande [20,60], jours >60 et >80, "
                "épisodes de stress consécutifs, impacts pluie/irrigation sur psi, "
                "et recommandations concrètes basées sur les valeurs."
                if current_lang == "fr"
                else "Analyze the chart data (psi, S, irrigation/rain, ETc, drainage). "
                     "Provide: % days in band [20,60], days >60 and >80, "
                     "consecutive stress episodes, rain/irrigation impacts on psi, "
                     "and concrete recommendations based on values."
            ),
            height=120,
            key="llm_prompt",
        )
        context_source_label = "Source d'analyse" if current_lang == "fr" else "Analysis source"
        context_source_options = []
        if st.session_state.get("evaluation_rollout") is not None:
            context_source_options.append("evaluation")
        if st.session_state.get("scenario1_result") is not None:
            context_source_options.append("scenario1")
        if not context_source_options:
            context_source_options = ["evaluation", "scenario1"]
        context_source_display = {
            "evaluation": "Évaluation (scénarios 2+)" if current_lang == "fr" else "Evaluation (scenarios 2+)",
            "scenario1": "Scénario 1 (règles)" if current_lang == "fr" else "Scenario 1 (rules)",
        }
        context_source = st.selectbox(
            context_source_label,
            options=context_source_options,
            format_func=lambda k: context_source_display.get(k, k),
            index=0,
            key="llm_context_source",
        )
        show_ctx_label = "Afficher le contexte" if current_lang == "fr" else "Show context"
        show_context = st.checkbox(show_ctx_label, value=False, key="llm_show_context")
        run_label = "Lancer" if current_lang == "fr" else "Run"
        if st.button(run_label, key="llm_run"):
            cfg = LLMConfig(
                backend=llm_backend,
                model=llm_model,
                endpoint=llm_endpoint or None,
                temperature=llm_temperature,
                max_tokens=int(llm_max_tokens),
            )
            llm = LocalLLM(cfg)
            with st.spinner("Génération..." if current_lang == "fr" else "Generating..."):
                try:
                    target_lang = current_lang if current_lang in {"fr", "en"} else "fr"
                    user_prompt = llm_prompt
                    if llm_translate_prompt:
                        src_lang = llm_prompt_lang
                        if src_lang == "auto":
                            src_lang = "fr" if target_lang == "en" else "en"
                        if src_lang != target_lang:
                            translation_system = (
                                "Traduire le texte utilisateur vers le français. "
                                "Répondre uniquement avec le texte traduit."
                                if target_lang == "fr"
                                else "Translate the user text to English. "
                                     "Respond only with the translated text."
                            )
                            translation = llm.generate(
                                user_prompt, system=translation_system
                            )
                            user_prompt = translation.text.strip()
                    system_prompt = llm_system
                    if llm_strict:
                        strict_rules = (
                            "Règles strictes: utilise uniquement les données fournies dans Context (JSON). "
                            "N'invente pas de variables, d'unités, ni d'événements. "
                            "Si une info manque, dis 'donnée non disponible'. "
                            "Réponds avec des faits chiffrés et cite les clés (ex: psi_cbar, I_mm, ETc_mm). "
                            "La 'tension' est la tension matricielle du sol (ψ) en cbar, pas électrique."
                            if current_lang == "fr"
                            else "Strict rules: use only the data provided in Context (JSON). "
                                 "Do not invent variables, units, or events. "
                                 "If information is missing, say 'data not available'. "
                                 "Answer with numeric facts and cite keys (e.g., psi_cbar, I_mm, ETc_mm). "
                                 "'Tension' refers to soil matric tension (ψ) in cbar, not electrical."
                        )
                        system_prompt = f"{system_prompt}\n\n{strict_rules}"
                    threshold_cbar = st.session_state.get("rule_threshold_cbar")
                    if threshold_cbar is None:
                        threshold_cbar = get_rule_seuil_unique_config().get("threshold_cbar")
                    assistant_context = {"threshold_cbar": threshold_cbar}
                    data_available = True
                    prompt_lower = user_prompt.lower()
                    inferred_scenario = None
                    if "scenario 1" in prompt_lower or "scénario 1" in prompt_lower:
                        inferred_scenario = "scenario1"
                    elif "scenario 2" in prompt_lower or "scénario 2" in prompt_lower:
                        inferred_scenario = "scenario2"
                    elif "scenario 3b" in prompt_lower or "scénario 3b" in prompt_lower:
                        inferred_scenario = "scenario3b"
                    elif "scenario 3" in prompt_lower or "scénario 3" in prompt_lower:
                        inferred_scenario = "scenario3"
                    elif "scenario 4" in prompt_lower or "scénario 4" in prompt_lower:
                        inferred_scenario = "scenario4"
                    elif "scenario 5" in prompt_lower or "scénario 5" in prompt_lower:
                        inferred_scenario = "scenario5"
                    elif "scenario 6" in prompt_lower or "scénario 6" in prompt_lower:
                        inferred_scenario = "scenario6"
                    assistant_context["assistant_inferred_scenario"] = inferred_scenario
                    if context_source == "scenario1" and ("scenario 2" in prompt_lower or "scénario 2" in prompt_lower):
                        st.warning(
                            "⚠️ Vous demandez le scénario 2, mais la source est réglée sur 'Scénario 1'."
                            if current_lang == "fr"
                            else "⚠️ You asked for scenario 2, but the source is set to 'Scenario 1'."
                        )
                        data_available = False
                    if context_source == "evaluation" and ("scenario 1" in prompt_lower or "scénario 1" in prompt_lower):
                        st.warning(
                            "⚠️ Vous demandez le scénario 1, mais la source est réglée sur 'Évaluation'."
                            if current_lang == "fr"
                            else "⚠️ You asked for scenario 1, but the source is set to 'Evaluation'."
                        )
                        data_available = False
                    if context_source == "evaluation" and st.session_state.get("evaluation_rollout") is None:
                        st.warning(
                            "⚠️ Aucune évaluation disponible. Lancez une évaluation d'abord."
                            if current_lang == "fr"
                            else "⚠️ No evaluation available. Run an evaluation first."
                        )
                        data_available = False
                    if context_source == "scenario1" and st.session_state.get("scenario1_result") is None:
                        st.warning(
                            "⚠️ Aucune simulation du scénario 1 disponible. Lancez la simulation d'abord."
                            if current_lang == "fr"
                            else "⚠️ No Scenario 1 simulation available. Run the simulation first."
                        )
                        data_available = False

                    def build_trend_summary(series: np.ndarray) -> Dict[str, float]:
                        if series.size < 2:
                            return {"slope_per_day": 0.0}
                        x = np.arange(series.size)
                        slope = float(np.polyfit(x, series, 1)[0])
                        return {"slope_per_day": slope}

                    def build_split_means(series: np.ndarray) -> Dict[str, float]:
                        if series.size == 0:
                            return {"mean_first_half": 0.0, "mean_second_half": 0.0}
                        mid = series.size // 2
                        return {
                            "mean_first_half": float(np.mean(series[:mid])) if mid > 0 else float(np.mean(series)),
                            "mean_second_half": float(np.mean(series[mid:])) if mid < series.size else float(np.mean(series)),
                        }

                    def build_series_pack(data: Dict[str, Any], key: str) -> Dict[str, Any]:
                        values = np.array(data.get(key, []), dtype=float)
                        if values.size == 0:
                            return {}
                        pack: Dict[str, Any] = {
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                        }
                        pack.update(build_trend_summary(values))
                        pack.update(build_split_means(values))
                        return pack

                    def build_detailed_rows(data: Dict[str, Any]) -> Dict[str, Any]:
                        psi = np.array(data.get("psi", []), dtype=float)
                        if psi.size == 0:
                            return {}
                        S = np.array(data.get("S", []), dtype=float)
                        I = np.array(data.get("I", []), dtype=float)
                        rain = np.array(data.get("rain", data.get("R", [])), dtype=float)
                        ET0 = np.array(data.get("ET0", []), dtype=float)
                        Kc = np.array(data.get("Kc", []), dtype=float)
                        ETc = np.array(data.get("ETc", []), dtype=float)
                        D = np.array(data.get("D", []), dtype=float)
                        rows = []
                        for i in range(int(psi.size)):
                            rows.append(
                                {
                                    "day": i + 1,
                                    "psi_cbar": float(psi[i]),
                                    "S_mm": float(S[i]) if i < S.size else None,
                                    "I_mm": float(I[i]) if i < I.size else None,
                                    "rain_mm": float(rain[i]) if i < rain.size else None,
                                    "ET0_mm": float(ET0[i]) if i < ET0.size else None,
                                    "Kc": float(Kc[i]) if i < Kc.size else None,
                                    "ETc_mm": float(ETc[i]) if i < ETc.size else None,
                                    "D_mm": float(D[i]) if i < D.size else None,
                                }
                                )
                        return {"n_days": int(psi.size), "rows": rows}

                    def compute_curve_insights(data: Dict[str, Any]) -> Dict[str, Any]:
                        psi = np.array(data.get("psi", []), dtype=float)
                        if psi.size == 0:
                            return {}
                        threshold = float(threshold_cbar)
                        band_low = 20.0
                        band_high = 60.0
                        total_days = int(psi.size)

                        def count_consecutive(mask: np.ndarray) -> Dict[str, Any]:
                            max_run = 0
                            run_lengths = []
                            run = 0
                            for is_on in mask:
                                if is_on:
                                    run += 1
                                else:
                                    if run > 0:
                                        run_lengths.append(run)
                                    max_run = max(max_run, run)
                                    run = 0
                            if run > 0:
                                run_lengths.append(run)
                            max_run = max(max_run, run)
                            return {
                                "max_run": int(max_run),
                                "runs_ge_3": int(sum(1 for r in run_lengths if r >= 3)),
                                "runs_ge_5": int(sum(1 for r in run_lengths if r >= 5)),
                            }

                        in_band = (psi >= band_low) & (psi <= band_high)
                        above_60 = psi > band_high
                        above_80 = psi > 80.0
                        below_20 = psi < band_low
                        above_threshold = psi > threshold

                        insights = {
                            "n_days": total_days,
                            "band_low_cbar": band_low,
                            "band_high_cbar": band_high,
                            "threshold_cbar": threshold,
                            "pct_in_band": float(np.mean(in_band) * 100.0),
                            "pct_above_60": float(np.mean(above_60) * 100.0),
                            "pct_above_80": float(np.mean(above_80) * 100.0),
                            "pct_below_20": float(np.mean(below_20) * 100.0),
                            "days_above_60": int(np.sum(above_60)),
                            "days_above_80": int(np.sum(above_80)),
                            "days_below_20": int(np.sum(below_20)),
                            "days_above_threshold": int(np.sum(above_threshold)),
                            "consecutive_stress_above_60": count_consecutive(above_60),
                            "consecutive_below_20": count_consecutive(below_20),
                        }

                        I = np.array(data.get("I", []), dtype=float)
                        rain = np.array(data.get("rain", data.get("R", [])), dtype=float)
                        ETc = np.array(data.get("ETc", []), dtype=float)
                        D = np.array(data.get("D", []), dtype=float)
                        if I.size == total_days:
                            insights["irrigation_events"] = int(np.sum(I > 0))
                            insights["mean_irrigation_mm"] = float(np.mean(I))
                            insights["total_irrigation_mm"] = float(np.sum(I))
                        if rain.size == total_days:
                            insights["rain_events"] = int(np.sum(rain > 0))
                            insights["mean_rain_mm"] = float(np.mean(rain))
                            insights["total_rain_mm"] = float(np.sum(rain))
                        if ETc.size == total_days:
                            insights["mean_ETc_mm"] = float(np.mean(ETc))
                            insights["total_ETc_mm"] = float(np.sum(ETc))
                        if D.size == total_days:
                            insights["mean_D_mm"] = float(np.mean(D))
                            insights["total_D_mm"] = float(np.sum(D))
                        return insights

                    if data_available:
                        source_label = (
                            "Scénario 1 (règles)" if context_source == "scenario1" else "Évaluation (scénarios 2+)"
                        )
                        user_prompt = f"[Source: {source_label}] {user_prompt}"
                        if context_source == "scenario1":
                            sim_result = st.session_state.get("scenario1_result")
                            if sim_result is not None and "psi" in sim_result:
                                psi = np.array(sim_result["psi"])
                                days = np.where(psi > float(threshold_cbar))[0] + 1
                                assistant_context["scenario1_days_above_threshold"] = {
                                    "count": int(days.size),
                                    "days": days[:30].tolist(),
                                    "truncated": days.size > 30,
                                }
                                assistant_context["scenario1_summary"] = {
                                    "n_days": int(psi.size),
                                    "psi_cbar": build_series_pack(sim_result, "psi"),
                                    "S_mm": build_series_pack(sim_result, "S"),
                                    "I_mm": build_series_pack(sim_result, "I"),
                                    "rain_mm": build_series_pack(sim_result, "rain"),
                                    "ETc_mm": build_series_pack(sim_result, "ETc"),
                                    "D_mm": build_series_pack(sim_result, "D"),
                                }
                                assistant_context["scenario1_detailed"] = build_detailed_rows(sim_result)
                                assistant_context["scenario1_curve_insights"] = compute_curve_insights(sim_result)
                        else:
                            rollout = st.session_state.get("evaluation_rollout")
                            if rollout is not None and "psi" in rollout:
                                psi = np.array(rollout["psi"])
                                days = np.where(psi > float(threshold_cbar))[0] + 1
                                assistant_context["eval_days_above_threshold"] = {
                                    "count": int(days.size),
                                    "days": days[:30].tolist(),
                                    "truncated": days.size > 30,
                                }
                                assistant_context["evaluation_summary"] = {
                                    "n_days": int(psi.size),
                                    "psi_cbar": build_series_pack(rollout, "psi"),
                                    "S_mm": build_series_pack(rollout, "S"),
                                    "I_mm": build_series_pack(rollout, "I"),
                                    "R_mm": build_series_pack(rollout, "R"),
                                    "ETc_mm": build_series_pack(rollout, "ETc"),
                                    "D_mm": build_series_pack(rollout, "D"),
                                }
                                assistant_context["evaluation_detailed"] = build_detailed_rows(rollout)
                                assistant_context["evaluation_curve_insights"] = compute_curve_insights(rollout)
                        assistant_context["units"] = {
                            "psi": "cbar",
                            "I": "mm",
                            "R": "mm",
                            "rain": "mm",
                            "ETc": "mm",
                            "D": "mm",
                        }
                        assistant_context["domain_glossary"] = {
                            "tension": "tension matricielle du sol (ψ) en cbar",
                            "S": "réserve en eau du sol (mm)",
                            "I": "irrigation (mm)",
                            "R/rain": "pluie (mm)",
                            "ETc": "évapotranspiration culturale (mm)",
                            "D": "drainage (mm)",
                        }
                        response = llm.generate(
                            user_prompt, system=system_prompt, context=assistant_context
                        )
                        st.success(response.text)
                        st.caption(
                            f"Modèle: {response.model} · {response.elapsed_s:.2f}s"
                            if current_lang == "fr"
                            else f"Model: {response.model} · {response.elapsed_s:.2f}s"
                        )
                        if show_context:
                            st.markdown("**Context (JSON)**")
                            st.json(assistant_context)
                except Exception as exc:
                    st.error(str(exc))
