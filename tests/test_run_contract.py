from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import streamlit_app as biotech_app
from valuation_codex_package.core import ModelConfig, Portfolio, Product, ProductConfig


def _install_fake_streamlit(monkeypatch, session_state: dict | None = None) -> dict:
    fake_state = session_state or {}
    monkeypatch.setattr(biotech_app, "st", SimpleNamespace(session_state=fake_state))
    return fake_state


def _sample_portfolio() -> tuple[ModelConfig, Portfolio]:
    model_cfg = ModelConfig(
        first_year=2024,
        n_years=6,
        currency="USD",
        discount_rate=0.1,
        tax_rate=0.25,
        ev_ebitda_multiple=8.0,
    )
    product_cfg = ProductConfig(
        name="ContractAsset",
        stage="Commercial",
        success_prob=1.0,
        include_in_consolidation=True,
        preexisting_market=True,
        time_to_market=0,
        patent_years=8,
        patent_revenue_target=100.0,
        post_patent_revenue_target=50.0,
    )
    return model_cfg, Portfolio([Product(product_cfg, model_cfg)], model_cfg)


def test_run_cached_base_valuation_reuses_same_result(monkeypatch) -> None:
    _install_fake_streamlit(monkeypatch, {})
    model_cfg, portfolio = _sample_portfolio()
    calls: list[str] = []

    class DummyEngine:
        def __init__(self, portfolio_obj):
            self.portfolio = portfolio_obj

        def run(self):
            calls.append("run")
            return {"portfolio_size": len(self.portfolio.products)}

    monkeypatch.setattr(biotech_app, "ValuationEngine", DummyEngine)

    signature_one, result_one = biotech_app._run_cached_base_valuation(model_cfg, portfolio)
    signature_two, result_two = biotech_app._run_cached_base_valuation(model_cfg, portfolio)

    assert signature_one == signature_two
    assert result_one is result_two
    assert calls == ["run"]


def test_lazy_analytics_helpers_mark_stale_after_signature_change(monkeypatch) -> None:
    _install_fake_streamlit(monkeypatch, {})

    result = biotech_app._run_cached_lazy_analytics(
        "scenario_section",
        base_signature="run-v1",
        params={"rev": 1.0},
        compute=lambda: {"rnpv": 123.0},
    )

    assert result == {"rnpv": 123.0}
    assert biotech_app._last_lazy_result("scenario_section") == {"rnpv": 123.0}
    assert (
        biotech_app._lazy_result_is_stale(
            "scenario_section",
            base_signature="run-v1",
            params={"rev": 1.0},
        )
        is False
    )
    assert (
        biotech_app._lazy_result_is_stale(
            "scenario_section",
            base_signature="run-v2",
            params={"rev": 1.0},
        )
        is True
    )


def test_get_state_prefers_draft_model_config(monkeypatch) -> None:
    fake_state = _install_fake_streamlit(
        monkeypatch,
        {
            "draft_model_config": ModelConfig(first_year=2030, n_years=9, currency="USD"),
            "model_config": ModelConfig(first_year=2024, n_years=5, currency="EUR"),
            "product_table": pd.DataFrame([{"name": "Asset A"}]),
            "planned_new_equity": 25.0,
        },
    )

    state = biotech_app.get_state()

    assert state["model_config"]["first_year"] == 2030
    assert state["planned_new_equity"] == 25.0
    assert "product_table" in state
    assert fake_state["model_config"].first_year == 2024


def test_set_state_restores_draft_only_and_clears_computed_outputs(monkeypatch) -> None:
    fake_state = _install_fake_streamlit(
        monkeypatch,
        {
            "model_config": ModelConfig(first_year=2024, n_years=5),
            "portfolio": object(),
            "valuation_result": object(),
            "biotech_last_run_signature": "run-v1",
            "financial_excel_bytes": b"cached",
        },
    )

    biotech_app.set_state(
        {
            "model_config": {"first_year": 2032, "n_years": 7, "currency": "USD"},
            "planned_new_equity": 55.0,
        }
    )

    assert fake_state["draft_model_config"].first_year == 2032
    assert fake_state["planned_new_equity"] == 55.0
    assert "model_config" not in fake_state
    assert "portfolio" not in fake_state
    assert "valuation_result" not in fake_state
    assert fake_state["biotech_results_stale"] is True
