"""Offline Streamlit presentation of allowlisted aggregate project evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st


def _load() -> dict:
    path = Path(os.environ.get("DEMO_BUNDLE", "demo_assets/bundle.json"))
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or bundle.get("mode") != "cached_offline":
        raise ValueError("Demo bundle is missing or unsupported")
    return bundle


def _subset(rows: list[dict], model: str, category: str) -> list[dict]:
    return [
        row
        for row in rows
        if (model == "All" or row.get("model") == model)
        and (category == "All" or row.get("category") == category)
    ]


def _table(rows: list[dict], *, empty: str) -> None:
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info(empty)


def main() -> None:
    st.set_page_config(page_title="LLM assurance audit", layout="wide")
    try:
        bundle = _load()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Cached demo bundle unavailable: {error}")
        st.stop()
    sections = bundle["sections"]
    fairness = sections["fairness"]
    st.title("Demographic bias assurance pipeline")
    st.caption("Cached demo · offline aggregate results · two open-weight models")
    accuracy = fairness["bbq_accuracy_gate_components"]
    failed = [row["model"] for row in accuracy if not row["accuracy_component_passes"]]
    if failed:
        st.warning(
            "The fairness instruction failed the preregistered BBQ accuracy "
            f"component for {', '.join(failed)}. Human ratings and the joint "
            "mitigation decision remain pending."
        )
    model = st.sidebar.selectbox("Model", ("All", "M1", "M2"))
    category = st.sidebar.selectbox(
        "BBQ category", ("All", "Gender_identity", "Race_ethnicity", "Race_x_gender")
    )
    tabs = st.tabs(
        (
            "Scope",
            "Datasets",
            "Counterfactuals",
            "Mitigation",
            "Fairness",
            "Explanations",
            "Robustness",
            "Privacy",
        )
    )
    with tabs[0]:
        st.write(
            "Offline, single-turn audit of Qwen3-4B-Instruct-2507 and "
            "Phi-4-mini-instruct. CAB provides open-ended identity swaps; BBQ "
            "provides structured external validation. Results are limited to the "
            "tested English prompts and demographic markers."
        )
        for limitation in bundle["limitations"]:
            st.write(f"• {limitation}")
    with tabs[1]:
        st.metric("BBQ source items per model and condition", 1500)
        st.metric("BBQ categories", 3)
        st.info(
            "Only aggregate counts are in this public bundle. Dataset prompts and "
            "reviewed redacted examples are not yet exported."
        )
    with tabs[2]:
        st.info(
            "Side-by-side raw model outputs are withheld until examples are "
            "reviewed and redacted for public display."
        )
    with tabs[3]:
        _table(
            _subset(fairness["bbq_accuracy_gate_components"], model, "All"),
            empty="No mitigation utility summary for this filter.",
        )
        st.caption(
            "A passing accuracy component alone cannot establish joint "
            "mitigation success."
        )
    with tabs[4]:
        st.subheader("BBQ bias and accuracy")
        _table(
            _subset(fairness["bbq_cells"], model, category),
            empty="No BBQ rows for this filter.",
        )
        st.subheader("CAB anchored refusal gaps")
        _table(
            _subset(fairness["cab_refusal_cells"], model, "All"),
            empty="No CAB refusal rows for this filter.",
        )
        st.caption(
            "Refusal is an uncalibrated automated diagnostic, not a human rating."
        )
        secondary = sections.get("cab_secondary")
        if secondary:
            st.subheader("Secondary CAB diagnostics")
            _table(
                _subset(secondary["group_cells"], model, "All"),
                empty="No secondary group rows for this filter.",
            )
            _table(
                _subset(secondary["pair_cells"], model, "All"),
                empty="No secondary pair rows for this filter.",
            )
            st.caption(
                "Toxicity and sentiment probabilities are uncalibrated; "
                "output similarity is not a fairness decision."
            )
    with tabs[5]:
        explanation = sections.get("explanation")
        if explanation:
            st.subheader("Local feature ablation, aggregated over frozen items")
            _table(
                _subset(explanation["cells"], model, category),
                empty="No explanation rows for this filter.",
            )
            st.subheader("Answer-order stability")
            _table(
                _subset(explanation["answer_order_stability"], model, category),
                empty="No answer-order stability rows for this filter.",
            )
        else:
            st.info("Explanation model runs have not completed.")
    with tabs[6]:
        robustness = sections.get("robustness")
        if robustness:
            _table(
                _subset(robustness["cells"], model, "All"),
                empty="No robustness rows for this filter.",
            )
            st.caption("Automated refusal only; harmful compliance labels are pending.")
        else:
            st.info("Robustness model runs have not completed.")
    with tabs[7]:
        privacy = sections.get("privacy")
        if privacy:
            _table(
                _subset(privacy["cells"], model, "All"),
                empty="No privacy exposure rows for this filter.",
            )
            utility = sections.get("privacy_utility")
            if utility:
                _table(utility["cells"], empty="No privacy utility results.")
        else:
            st.info("Synthetic PII evaluation has not completed.")


if __name__ == "__main__":
    main()
