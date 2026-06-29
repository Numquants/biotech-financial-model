import unittest
from io import BytesIO
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    _apply_debt_schedule,
    _build_bankable_snapshot_payload,
    _build_stage_mapping_candidate_row,
    _build_financial_excel,
    _build_enterprise_to_equity_bridge,
    _build_investor_waterfall,
    _build_lender_metrics,
    _build_portfolio,
    _build_probability_preview,
    _default_stage_schedule_mapping,
    _stage_mapping_row_warnings,
    _set_dataframe_cell,
)
from valuation_codex_package.core import ModelConfig, Product, ProductConfig, Portfolio, ValuationEngine


class StreamlitHelperTests(unittest.TestCase):
    def test_set_dataframe_cell_upcasts_for_incompatible_editor_value(self) -> None:
        df = pd.DataFrame({"Year": [2025]})

        _set_dataframe_cell(df, 0, "Year", None)

        self.assertIsNone(df.at[0, "Year"])
        self.assertEqual(str(df["Year"].dtype), "object")

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

    def test_build_revenue_series_handles_readonly_series_values(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=6)
        product_cfg = ProductConfig(
            name="ReadonlyAsset",
            stage="Commercial",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=50.0,
            market_growth_patent=0.0,
            market_growth_post=0.0,
        )
        product = Product(product_cfg, model_cfg)

        def _readonly_values(series: pd.Series):
            values = series.to_numpy(copy=False)
            values.setflags(write=False)
            return values

        with patch.object(pd.Series, "values", new=property(_readonly_values)):
            revenue = product.build_revenue_series()

        self.assertEqual(len(revenue), model_cfg.n_years)
        self.assertTrue((revenue >= 0.0).all())

    def test_bankable_snapshot_payload_ignores_advisory_metric_overrides(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=5, discount_rate=0.10, tax_rate=0.20)
        product_cfg = ProductConfig(
            name="LockedSnapshotAsset",
            stage="Commercial",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        portfolio = Portfolio([Product(product_cfg, model_cfg)], model_cfg)
        valuation_result = ValuationEngine(portfolio).run()
        advisory_inputs = {
            "npv": 999_999_999.0,
            "irr": 9.99,
            "dscr_min": 0.1,
            "scenarios": [{"name": "Manual override", "npv": -123.0, "irr": -0.5}],
            "notes": "Do not export this.",
        }

        payload = _build_bankable_snapshot_payload(
            "BIO-LOCK",
            model_cfg,
            valuation_result,
            portfolio,
            workbook_hash="hash-123",
            advisory_inputs=advisory_inputs,
        )

        self.assertEqual(payload["snapshot_source"], "live_model_only")
        self.assertEqual(payload["project_id"], "BIO-LOCK")
        self.assertEqual(payload["workbook_hash"], "hash-123")
        self.assertEqual(payload["advisory_inputs"]["npv"], advisory_inputs["npv"])
        self.assertEqual(payload["advisory_inputs"]["notes"], advisory_inputs["notes"])
        self.assertNotEqual(payload["financial_snapshot"]["npv"], advisory_inputs["npv"])
        self.assertNotEqual(payload["financial_snapshot"]["irr"], advisory_inputs["irr"])
        self.assertNotEqual(payload["financial_snapshot"]["dscr_min"], advisory_inputs["dscr_min"])
        self.assertNotEqual(
            payload["financial_snapshot"]["scenarios"],
            advisory_inputs["scenarios"],
        )

    def test_streamlit_app_smoke_renders_without_exceptions(self) -> None:
        app = AppTest.from_file("streamlit_app.py")

        app.run(timeout=120)

        self.assertEqual(len(app.exception), 0, app.exception)

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

    def test_stage_mapping_candidate_row_derives_time_to_market_from_durations(self) -> None:
        mapping_df = _default_stage_schedule_mapping().copy()
        base_row = mapping_df.loc[mapping_df["Stage"] == "Phase II"].iloc[0]

        candidate = _build_stage_mapping_candidate_row(
            base_row,
            {
                "Stage": "Phase II",
                "Time to market (years)": 99,
                "Phase II duration (years)": 3,
                "Phase III duration (years)": 2,
                "Approval duration (years)": 1,
            },
        )

        self.assertEqual(candidate["Time to market (years)"], 6)

    def test_stage_mapping_row_warnings_preview_inconsistent_approval_row(self) -> None:
        mapping_df = _default_stage_schedule_mapping().copy()
        row_idx = mapping_df.index[mapping_df["Stage"] == "Approval"][0]
        base_row = mapping_df.loc[row_idx]

        candidate = _build_stage_mapping_candidate_row(
            base_row,
            {
                "Stage": "Approval",
                "Discovery duration (years)": 1,
                "Preclinical duration (years)": 1,
                "Phase I duration (years)": 1,
                "Phase II duration (years)": 2,
                "Phase III duration (years)": 1,
                "Approval duration (years)": 1,
            },
        )
        warnings = _stage_mapping_row_warnings(mapping_df, int(row_idx), candidate)

        self.assertTrue(any("time-to-market" in warning for warning in warnings))
        self.assertTrue(any("early-stage durations" in warning for warning in warnings))

    def test_debt_schedule_builds_bullet_profile_and_lender_metrics(self) -> None:
        cash_flow_df = pd.DataFrame(
            {
                "Net cash from operations": [80.0, 90.0, 100.0],
                "Net cash from investing": [-10.0, -10.0, -10.0],
                "Equity issuance": [0.0, 0.0, 0.0],
                "Net cash from financing": [0.0, 0.0, 0.0],
                "Net change in cash": [70.0, 80.0, 90.0],
                "Beginning cash balance": [0.0, 0.0, 0.0],
                "Ending cash balance": [70.0, 150.0, 240.0],
            },
            index=[2024, 2025, 2026],
        )
        debt_schedule = pd.DataFrame(
            {
                "Year": [2024, 2025, 2026],
                "Debt drawdowns": [120.0, 0.0, 0.0],
                "Manual debt repayments": [0.0, 0.0, 0.0],
            }
        )

        updated = _apply_debt_schedule(
            cash_flow_df,
            debt_schedule,
            interest_rate=0.10,
            repayment_mode="bullet",
            grace_years=1,
            minimum_cash_reserve=5.0,
        )
        self.assertEqual(updated.loc[2024, "Debt repayments"], 0.0)
        self.assertEqual(updated.loc[2025, "Debt repayments"], 0.0)
        self.assertEqual(updated.loc[2026, "Debt repayments"], 120.0)
        self.assertEqual(updated.loc[2025, "Beginning cash balance"], updated.loc[2024, "Ending cash balance"])

        metrics = _build_lender_metrics(
            updated,
            discount_rate=0.10,
            minimum_cash_reserve=5.0,
            target_dscr=1.2,
        )
        self.assertAlmostEqual(metrics.loc[2025, "DSCR"], 80.0 / 12.0)
        self.assertAlmostEqual(metrics.loc[2026, "Debt service"], 132.0)
        self.assertEqual(metrics.loc[2025, "Covenant status"], "Pass")

    def test_investor_waterfall_applies_preference_before_common(self) -> None:
        shareholders_df = pd.DataFrame(
            [
                {
                    "Shareholder": "Founders",
                    "Security": "Common",
                    "Seniority": 3,
                    "Ownership %": 0.5,
                    "Investment": 20.0,
                    "Liquidation preference (x)": 0.0,
                    "Participating preferred": False,
                },
                {
                    "Shareholder": "Series A",
                    "Security": "Preferred",
                    "Seniority": 1,
                    "Ownership %": 0.5,
                    "Investment": 40.0,
                    "Liquidation preference (x)": 1.0,
                    "Participating preferred": False,
                },
            ]
        )

        waterfall = _build_investor_waterfall(shareholders_df, exit_equity_value=60.0).set_index("Shareholder")
        self.assertEqual(waterfall.loc["Series A", "Decision"], "Take preference")
        self.assertEqual(waterfall.loc["Series A", "Preference paid"], 40.0)
        self.assertEqual(waterfall.loc["Founders", "Total proceeds"], 20.0)

    def test_financial_excel_includes_debt_and_waterfall_sheets(self) -> None:
        cons = pd.DataFrame(
            {
                "revenue": [100.0],
                "cogs": [-40.0],
                "ebitda": [20.0],
                "nopat": [12.0],
                "rd_cash": [-5.0],
                "capex_cash": [-3.0],
                "fcff_after_wc": [10.0],
            },
            index=[2024],
        )
        perf_df = pd.DataFrame({"Revenue": [100.0]}, index=[2024])
        position_df = pd.DataFrame({"Total equity": [50.0]}, index=[2024])
        cash_flow_df = pd.DataFrame({"Net change in cash": [10.0]}, index=[2024])
        lender_metrics = pd.DataFrame(
            {"CFADS": [10.0], "Debt service": [5.0], "DSCR": [2.0]},
            index=[2024],
        )
        investor_waterfall = pd.DataFrame(
            {
                "Shareholder": ["Founders"],
                "Total proceeds": [25.0],
            }
        )

        workbook = load_workbook(
            BytesIO(
                _build_financial_excel(
                    cons,
                    perf_df,
                    position_df,
                    cash_flow_df,
                    ModelConfig(first_year=2024, n_years=1),
                    lender_metrics=lender_metrics,
                    investor_waterfall=investor_waterfall,
                )
            )
        )
        self.assertIn("Debt metrics", workbook.sheetnames)
        self.assertIn("Investor waterfall", workbook.sheetnames)


if __name__ == "__main__":
    unittest.main()
