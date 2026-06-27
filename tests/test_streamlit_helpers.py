import unittest

import pandas as pd

from streamlit_app import (
    _build_enterprise_to_equity_bridge,
    _build_portfolio,
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

    def test_stage_generated_milestones_do_not_embed_transition_probability_twice(self) -> None:
        mapping_df = _default_stage_schedule_mapping().copy()
        phase_mask = mapping_df["Stage"] == "Phase II"
        mapping_df.loc[phase_mask, "Phase II duration (years)"] = 1
        mapping_df.loc[phase_mask, "Phase III duration (years)"] = 1
        mapping_df.loc[phase_mask, "Approval duration (years)"] = 0
        mapping_df.loc[phase_mask, "Time to market (years)"] = 2
        mapping_df.loc[phase_mask, "Phase II->Phase III"] = 50.0
        mapping_df.loc[phase_mask, "Phase III->Approval"] = 50.0
        mapping_df.loc[phase_mask, "Phase II->Phase III annual success %"] = 50.0
        mapping_df.loc[phase_mask, "Phase III->Approval annual success %"] = 50.0
        for col in [c for c in mapping_df.columns if c.endswith("completion milestone (USD)")]:
            mapping_df.loc[phase_mask, col] = 0.0
        mapping_df.loc[phase_mask, "Phase II completion milestone (USD)"] = 100.0

        product_df = pd.DataFrame(
            [
                {
                    "name": "StageMilestone",
                    "stage": "Phase II",
                    "success_prob": 0.9,
                    "include_in_consolidation": True,
                    "time_to_market": 2,
                    "patent_years": 5,
                    "patent_revenue_target": 0.0,
                    "post_patent_revenue_target": 0.0,
                }
            ]
        )
        model_cfg = ModelConfig(
            first_year=2024,
            n_years=10,
            discount_rate=0.0,
            tax_rate=0.0,
            ev_ebitda_multiple=0.0,
        )
        portfolio = _build_portfolio(
            product_df,
            model_cfg,
            stage_mapping=mapping_df,
            overwrite_defaults=False,
            detail_tables={},
        )
        self.assertIsNotNone(portfolio)
        product = portfolio.products[0]
        normalized_milestones = product._normalized_milestones()
        self.assertEqual(len(normalized_milestones), 1)
        self.assertEqual(normalized_milestones[0].probability, 1.0)

        valuation_result = ValuationEngine(portfolio).run()
        milestone_year = model_cfg.first_year + normalized_milestones[0].year_offset
        expected_weighted_milestone = (
            float(normalized_milestones[0].amount)
            * float(product._success_prob_schedule().loc[milestone_year])
        )
        self.assertEqual(valuation_result.per_product["StageMilestone"].loc[milestone_year, "milestones"], 100.0)
        self.assertEqual(
            valuation_result.consolidated.loc[milestone_year, "milestones"],
            expected_weighted_milestone,
        )


if __name__ == "__main__":
    unittest.main()
