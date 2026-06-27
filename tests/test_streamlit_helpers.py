import unittest

import pandas as pd

from streamlit_app import (
    _build_enterprise_to_equity_bridge,
    _build_probability_preview,
    _default_stage_schedule_mapping,
)
from valuation_codex_package.core import ModelConfig, Product, ProductConfig, Portfolio, ValuationEngine


class StreamlitHelperTests(unittest.TestCase):
    def test_enterprise_to_equity_bridge_applies_cash_debt_and_new_equity(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=1, discount_rate=0.0, tax_rate=0.0, ev_ebitda_multiple=0.0)
        product_cfg = ProductConfig(
            name="BridgeAsset",
            stage="Commercial",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        valuation_result = ValuationEngine(Portfolio([Product(product_cfg, model_cfg)], model_cfg)).run()
        cash_flow_df = pd.DataFrame(
            {
                "Ending cash balance": [15.0],
                "Debt closing balance": [40.0],
            },
            index=[2024],
        )

        bridge = _build_enterprise_to_equity_bridge(valuation_result, cash_flow_df, planned_new_equity=25.0)
        amounts = dict(zip(bridge["Component"], bridge["Amount"]))
        self.assertEqual(amounts["Enterprise value (DCF)"], valuation_result.enterprise_value)
        self.assertEqual(amounts["Less: debt outstanding"], -40.0)
        self.assertEqual(amounts["Add: cash / (cash deficit)"], 15.0)
        self.assertEqual(amounts["Post-money equity value"], amounts["Pre-money equity value"] + 25.0)

    def test_probability_preview_flags_stage_transition_source(self) -> None:
        product_df = pd.DataFrame(
            [
                {
                    "name": "AgSeed-101",
                    "stage": "Phase 2",
                    "success_prob": 0.35,
                    "include_in_consolidation": True,
                    "time_to_market": 4,
                    "patent_years": 15,
                    "patent_revenue_target": 120_000_000,
                    "post_patent_revenue_target": 60_000_000,
                }
            ]
        )
        preview = _build_probability_preview(
            product_df,
            ModelConfig(first_year=2024, n_years=10),
            _default_stage_schedule_mapping(),
            overwrite_defaults=False,
            detail_tables={},
        )
        self.assertEqual(preview.iloc[0]["Stage"], "Phase II")
        self.assertEqual(preview.iloc[0]["Probability source"], "Stage-transition path")
        self.assertEqual(preview.iloc[0]["Probability path used"], "Stage-transition path")
        self.assertGreater(preview.iloc[0]["Effective cumulative success probability"], 0.0)

    def test_probability_preview_normalizes_market_alias_to_commercial(self) -> None:
        product_df = pd.DataFrame(
            [
                {
                    "name": "CommercialAsset",
                    "stage": "Market",
                    "success_prob": 1.0,
                    "include_in_consolidation": True,
                    "time_to_market": 0,
                    "patent_years": 10,
                    "patent_revenue_target": 50_000_000,
                    "post_patent_revenue_target": 25_000_000,
                }
            ]
        )
        preview = _build_probability_preview(
            product_df,
            ModelConfig(first_year=2024, n_years=10),
            _default_stage_schedule_mapping(),
            overwrite_defaults=False,
            detail_tables={},
        )
        self.assertEqual(preview.iloc[0]["Stage"], "Commercial")
        self.assertEqual(preview.iloc[0]["Probability source"], "Stage-transition path")


if __name__ == "__main__":
    unittest.main()
