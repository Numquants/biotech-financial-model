import unittest

import pandas as pd

from streamlit_app import (
    _apply_debt_schedule,
    _build_portfolio,
    _compute_financial_statements,
    _compute_irr,
    _default_debt_schedule,
    _default_market_share_table,
    _default_market_size_estimation_table,
    _default_products,
    _default_ramp_schedule,
    _default_royalty_table,
    _default_stage_schedule_mapping,
    _default_vaccine_capex_table,
    _default_vaccine_cost_table,
    _default_vaccine_development_table,
    _default_vaccine_rd_table,
    _default_vaccine_revenue_table,
    _default_vaccine_sales_table,
    _sanitize_product_records,
)
from valuation_codex_package.core import ModelConfig, ValuationEngine, validate_portfolio


def _default_model_config() -> ModelConfig:
    return ModelConfig(
        first_year=2024,
        n_years=25,
        currency="USD",
        discount_rate=0.10,
        tax_rate=0.25,
        working_capital_pct_sales=0.08,
        ev_ebitda_multiple=8.0,
        sales_ramp_factors=_default_ramp_schedule()["Ramp factor"].astype(float).tolist(),
    )


def _cashflows_with_terminal(result) -> list[float]:
    cashflows = result.dcf_table["fcff"].tolist()
    cashflows[-1] += float(result.dcf_table["terminal_value"].fillna(0.0).iloc[-1])
    if "working_capital_recovery" in result.dcf_table:
        cashflows[-1] += float(result.dcf_table["working_capital_recovery"].fillna(0.0).iloc[-1])
    return cashflows


class DefaultBaselineTests(unittest.TestCase):
    def test_default_product_baseline_is_profitable_and_financed(self) -> None:
        model_cfg = _default_model_config()
        portfolio = _build_portfolio(
            _default_products(),
            model_cfg,
            stage_mapping=_default_stage_schedule_mapping(),
            overwrite_defaults=False,
            detail_tables={},
        )

        result = ValuationEngine(portfolio).run()
        cons = result.consolidated
        _, _, cash_flow_df = _compute_financial_statements(cons, model_cfg)
        financed_cash = _apply_debt_schedule(
            cash_flow_df,
            _default_debt_schedule(model_cfg.first_year, model_cfg.n_years),
            0.08,
        )
        irr = _compute_irr(_cashflows_with_terminal(result))

        self.assertGreater(result.rnpv, 0.0)
        self.assertIsNotNone(irr)
        self.assertGreater(irr, 0.0)
        self.assertGreater(result.dcf_table["terminal_value"].fillna(0.0).iloc[-1], 0.0)
        self.assertLess(cons["fcff_after_wc"].iloc[0], 0.0)
        self.assertGreater(cons["fcff_after_wc"].iloc[-1], 0.0)
        self.assertGreater(financed_cash["Ending cash balance"].min(), 0.0)
        self.assertAlmostEqual(financed_cash["Debt closing balance"].iloc[-1], 0.0, places=6)

    def test_default_product_baseline_passes_validation(self) -> None:
        model_cfg = _default_model_config()
        portfolio = _build_portfolio(
            _default_products(),
            model_cfg,
            stage_mapping=_default_stage_schedule_mapping(),
            overwrite_defaults=False,
            detail_tables={},
        )

        self.assertEqual(validate_portfolio(portfolio), [])

    def test_detailed_default_baseline_stays_profitable(self) -> None:
        model_cfg = _default_model_config()
        detail_tables = {
            "development": _default_vaccine_development_table(model_cfg.first_year),
            "market_size_estimation": _default_market_size_estimation_table(),
            "revenue": _default_vaccine_revenue_table(),
            "cost": _default_vaccine_cost_table(),
            "rd": _default_vaccine_rd_table(),
            "capex": _default_vaccine_capex_table(),
            "royalty": _default_royalty_table(),
            "market_share": _default_market_share_table(),
        }
        portfolio = _build_portfolio(
            _default_products(),
            model_cfg,
            stage_mapping=_default_stage_schedule_mapping(),
            overwrite_defaults=False,
            detail_tables=detail_tables,
        )

        result = ValuationEngine(portfolio).run()
        cons = result.consolidated
        _, _, cash_flow_df = _compute_financial_statements(cons, model_cfg)
        financed_cash = _apply_debt_schedule(
            cash_flow_df,
            _default_debt_schedule(model_cfg.first_year, model_cfg.n_years),
            0.08,
        )
        irr = _compute_irr(_cashflows_with_terminal(result))

        self.assertGreater(result.rnpv, 0.0)
        self.assertIsNotNone(irr)
        self.assertGreater(irr, 0.0)
        self.assertLess(cons["fcff_after_wc"].iloc[0], 0.0)
        self.assertGreater(cons["fcff_after_wc"].iloc[-1], 0.0)
        self.assertGreater(result.dcf_table["terminal_value"].fillna(0.0).iloc[-1], 0.0)
        self.assertGreater(financed_cash["Ending cash balance"].min(), 0.0)

    def test_default_vaccine_sales_match_seed_revenue_targets(self) -> None:
        sales_df = _default_vaccine_sales_table(2024, 5)
        revenue_df = _default_vaccine_revenue_table().copy()
        revenue_df["Target revenue"] = (
            pd.to_numeric(revenue_df["Patent customers per year"], errors="coerce").fillna(0.0)
            * pd.to_numeric(revenue_df["Patent price (USD/customer)"], errors="coerce").fillna(0.0)
        )
        implied = (
            sales_df.assign(
                implied_revenue=
                pd.to_numeric(sales_df["Doses (M)"], errors="coerce").fillna(0.0)
                * 1e6
                * pd.to_numeric(sales_df["Price per dose"], errors="coerce").fillna(0.0)
            )
            .groupby("ID_vaccine", as_index=False)["implied_revenue"]
            .mean()
        )
        merged = implied.merge(
            revenue_df[["ID_vaccine", "Target revenue"]],
            on="ID_vaccine",
            how="inner",
        )

        self.assertFalse(merged.empty)
        for _, row in merged.iterrows():
            self.assertAlmostEqual(row["implied_revenue"], row["Target revenue"], places=2)

    def test_late_stage_rd_defaults_allocate_to_remaining_phases_only(self) -> None:
        model_cfg = _default_model_config()
        detail_tables = {
            "development": _default_vaccine_development_table(model_cfg.first_year),
            "market_size_estimation": _default_market_size_estimation_table(),
            "revenue": _default_vaccine_revenue_table(),
            "cost": _default_vaccine_cost_table(),
            "rd": _default_vaccine_rd_table(),
            "capex": _default_vaccine_capex_table(),
            "royalty": _default_royalty_table(),
            "market_share": _default_market_share_table(),
        }
        records = _sanitize_product_records(
            _default_products(),
            stage_mapping=_default_stage_schedule_mapping(),
            overwrite_defaults=False,
            detail_tables=detail_tables,
        )
        agseed = next(record for record in records if record["name"] == "AgSeed-101")

        self.assertEqual(agseed["stage"], "Approval")
        self.assertEqual(set(agseed["trial_costs_by_phase"]), {"Approval"})
        self.assertAlmostEqual(
            sum(agseed["trial_costs_by_phase"].values()),
            float(agseed["rd_remaining_pre_launch"]),
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
