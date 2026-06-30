import unittest

import numpy as np

from valuation_codex_package.core import (
    ModelConfig,
    Milestone,
    ProductConfig,
    Product,
    Portfolio,
    ValuationEngine,
    Scenario,
    ScenarioEngine,
    MonteCarloEngine,
    validate_portfolio,
    validate_product_config,
)


class CoreModelTests(unittest.TestCase):
    def test_revenue_series_preexisting_market_respects_ramp(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=5, sales_ramp_factors=[0.5, 1.0, 1.0, 1.0, 1.0])
        product_cfg = ProductConfig(
            name="TestProduct",
            stage="Market",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=3,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        product = Product(product_cfg, model_cfg)

        revenue = product.build_revenue_series()
        expected = np.array([50.0, 100.0, 100.0, 100.0, 100.0])
        np.testing.assert_allclose(revenue.values, expected)

    def test_valuation_engine_zero_cashflows(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=3)
        product_cfg = ProductConfig(
            name="ZeroProduct",
            stage="Discovery",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=False,
            time_to_market=3,
            patent_years=5,
            patent_revenue_target=0.0,
            post_patent_revenue_target=0.0,
        )
        product = Product(product_cfg, model_cfg)
        portfolio = Portfolio([product], model_cfg)

        result = ValuationEngine(portfolio).run()
        self.assertEqual(result.rnpv, 0.0)
        self.assertTrue((result.consolidated["fcff_after_wc"] == 0.0).all())

    def test_success_prob_schedule_clamps_to_horizon(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=3)
        product_cfg = ProductConfig(
            name="ClampProduct",
            stage="Discovery",
            success_prob=0.4,
            include_in_consolidation=True,
            preexisting_market=False,
            time_to_market=6,
            patent_years=5,
            patent_revenue_target=0.0,
            post_patent_revenue_target=0.0,
        )
        schedule = Product(product_cfg, model_cfg)._success_prob_schedule()
        self.assertEqual(len(schedule), 3)
        np.testing.assert_allclose(schedule.values, np.array([1.0, 0.9, 0.8]))

    def test_milestones_not_double_counted_and_weighted_once(self) -> None:
        model_cfg = ModelConfig(
            first_year=2024,
            n_years=1,
            tax_rate=0.0,
            discount_rate=0.0,
            ev_ebitda_multiple=0.0,
        )
        product_cfg = ProductConfig(
            name="MilestoneProduct",
            stage="Commercial",
            success_prob=0.5,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=0.0,
            post_patent_revenue_target=0.0,
            milestones=[Milestone(name="approval", year_offset=0, amount=100.0, probability=0.2, timing="from_start")],
        )
        result = ValuationEngine(Portfolio([Product(product_cfg, model_cfg)], model_cfg)).run()
        self.assertEqual(result.consolidated.loc[2024, "milestones"], 10.0)
        self.assertEqual(result.consolidated.loc[2024, "fcff"], 10.0)
        self.assertEqual(result.rnpv, 10.0)

    def test_stage_transition_probabilities_override_success_prob(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=5)
        product_cfg = ProductConfig(
            name="StageDriven",
            stage="Phase 2",
            success_prob=0.9,
            include_in_consolidation=True,
            time_to_market=2,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
            stage_duration_years={"Phase II": 1, "Phase III": 1, "Approval": 0},
            stage_transition_probabilities={"Phase II->Phase III": 0.5, "Phase III->Approval": 0.5},
            stage_transition_curve={"Phase II->Phase III": [0.5], "Phase III->Approval": [0.5]},
        )
        product = Product(product_cfg, model_cfg)
        self.assertEqual(product.config.stage, "Phase II")
        self.assertAlmostEqual(product.effective_success_probability(), 0.25)
        self.assertEqual(product.probability_source(), "stage_transitions")

    def test_validate_product_config_rejects_impossible_operating_ratios(self) -> None:
        issues = validate_product_config(
            ProductConfig(
                name="InvalidRatios",
                stage="Commercial",
                success_prob=1.0,
                preexisting_market=True,
                time_to_market=0,
                cogs_patent=1.2,
                cogs_post=-0.1,
                labor_pct=1.1,
                overhead_pct=-0.2,
                royalty_pct=1.5,
                post_patent_erosion=[1.0, 1.1],
            )
        )
        joined = " | ".join(issues)
        self.assertIn("cogs_patent must be between 0 and 1", joined)
        self.assertIn("cogs_post must be between 0 and 1", joined)
        self.assertIn("labor_pct must be between 0 and 1", joined)
        self.assertIn("overhead_pct must be between 0 and 1", joined)
        self.assertIn("royalty_pct must be between 0 and 1", joined)
        self.assertIn("post_patent_erosion values must be between 0 and 1", joined)

    def test_validate_product_config_rejects_inconsistent_stage_and_revenue_inputs(self) -> None:
        issues = validate_product_config(
            ProductConfig(
                name="InconsistentAsset",
                stage="Phase II",
                success_prob=0.4,
                include_in_consolidation=True,
                time_to_market=3,
                patent_years=10,
                patent_revenue_target=100.0,
                post_patent_revenue_target=50.0,
                patient_population_patent=2.0,
                price_per_patient_patent=40.0,
                penetration_patent=1.0,
                stage_duration_years={"Phase II": 1, "Phase III": 1, "Approval": 0},
                stage_cost_weights={"Phase II": 0.7, "Phase III": 0.6},
                stage_capex_weights={"Phase II": 0.6, "Phase III": 0.6},
            )
        )
        joined = " | ".join(issues)
        self.assertIn("patient-based patent-period revenue does not reconcile", joined)
        self.assertIn("stage durations from Phase II to launch must sum to time_to_market", joined)
        self.assertNotIn("stage cost weights across active pre-launch stages", joined)
        self.assertNotIn("stage capex weights across active pre-launch stages", joined)

    def test_validate_product_config_rejects_missing_active_stage_weights(self) -> None:
        issues = validate_product_config(
            ProductConfig(
                name="MissingWeightsAsset",
                stage="Approval",
                success_prob=0.8,
                include_in_consolidation=True,
                time_to_market=1,
                patent_years=10,
                patent_revenue_target=100.0,
                post_patent_revenue_target=50.0,
                stage_duration_years={"Approval": 1},
                stage_cost_weights={"Approval": 0.0},
                stage_capex_weights={"Approval": 0.0},
            )
        )
        joined = " | ".join(issues)
        self.assertIn("stage cost weights across active pre-launch stages must allocate some positive weight", joined)
        self.assertIn("stage capex weights across active pre-launch stages must allocate some positive weight", joined)

    def test_validate_portfolio_rejects_out_of_range_model_assumptions(self) -> None:
        model_cfg = ModelConfig(
            first_year=2024,
            n_years=5,
            tax_rate=0.8,
            discount_rate=0.04,
            ev_ebitda_multiple=35.0,
            working_capital_pct_sales=1.2,
            terminal_method="perpetuity_growth",
            perpetuity_growth_rate=0.05,
        )
        product_cfg = ProductConfig(
            name="ModelIssue",
            stage="Commercial",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=50.0,
        )
        issues = validate_portfolio(Portfolio([Product(product_cfg, model_cfg)], model_cfg))
        joined = " | ".join(issues)
        self.assertIn("ModelConfig: tax_rate must be between 0 and 0.6", joined)
        self.assertIn("ModelConfig: ev_ebitda_multiple must be between 0 and 30", joined)
        self.assertIn("ModelConfig: working_capital_pct_sales must be between 0 and 1", joined)
        self.assertIn("ModelConfig: perpetuity_growth_rate must be lower than discount_rate", joined)

    def test_opening_nol_shields_consolidated_tax(self) -> None:
        model_cfg = ModelConfig(
            first_year=2024,
            n_years=1,
            tax_rate=0.2,
            discount_rate=0.0,
            ev_ebitda_multiple=0.0,
            opening_nol_balance=50.0,
            sales_ramp_factors=[1.0],
        )
        product_cfg = ProductConfig(
            name="Taxable",
            stage="Commercial",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
            cogs_patent=0.0,
            cogs_post=0.0,
            labor_pct=0.0,
            overhead_pct=0.0,
            material_pct=0.0,
            sales_marketing_pct=0.0,
            gna_pct=0.0,
        )
        result = ValuationEngine(Portfolio([Product(product_cfg, model_cfg)], model_cfg)).run()
        self.assertEqual(result.consolidated.loc[2024, "taxable_income_after_nol"], 50.0)
        self.assertEqual(result.consolidated.loc[2024, "tax"], -10.0)

    def test_year_0_discounting_starts_at_par(self) -> None:
        model_cfg = ModelConfig(
            first_year=2024,
            n_years=2,
            discount_rate=0.1,
            ev_ebitda_multiple=0.0,
            discount_timing="year_0",
        )
        product_cfg = ProductConfig(
            name="Par",
            stage="Commercial",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=100.0,
        )
        result = ValuationEngine(Portfolio([Product(product_cfg, model_cfg)], model_cfg)).run()
        self.assertEqual(result.dcf_table.iloc[0]["discount_factor"], 1.0)

    def test_perpetuity_terminal_and_wc_recovery_are_added(self) -> None:
        model_cfg = ModelConfig(
            first_year=2024,
            n_years=2,
            discount_rate=0.1,
            tax_rate=0.0,
            working_capital_pct_sales=0.1,
            terminal_method="perpetuity_growth",
            perpetuity_growth_rate=0.02,
        )
        product_cfg = ProductConfig(
            name="Terminal",
            stage="Commercial",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=10,
            patent_revenue_target=100.0,
            post_patent_revenue_target=100.0,
        )
        result = ValuationEngine(Portfolio([Product(product_cfg, model_cfg)], model_cfg)).run()
        self.assertGreater(result.dcf_table.loc[2025, "terminal_value"], 0.0)
        self.assertEqual(
            result.dcf_table.loc[2025, "working_capital_recovery"],
            result.consolidated.loc[2025, "working_capital_balance"],
        )

    def test_scenario_engine_applies_discount_shift(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=3)
        product_cfg = ProductConfig(
            name="ScenarioProduct",
            stage="Market",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        portfolio = Portfolio([Product(product_cfg, model_cfg)], model_cfg)

        scenario = Scenario(name="RateUp", discount_rate_shift=0.02)
        results = ScenarioEngine(portfolio).run_scenarios([scenario])
        rate_row = results.loc[results["scenario"] == "RateUp", "discount_rate"].iloc[0]
        self.assertAlmostEqual(rate_row, model_cfg.discount_rate + 0.02)

    def test_monte_carlo_reproducible_seed(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=3)
        product_cfg = ProductConfig(
            name="MCProduct",
            stage="Market",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        portfolio = Portfolio([Product(product_cfg, model_cfg)], model_cfg)
        engine = MonteCarloEngine(portfolio)

        sims_a = engine.simulate(n_sims=10, random_seed=42)
        sims_b = engine.simulate(n_sims=10, random_seed=42)
        np.testing.assert_allclose(sims_a.values, sims_b.values)

    def test_monte_carlo_launch_delay_is_reproducible(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=4)
        product_cfg = ProductConfig(
            name="MCLaunch",
            stage="Phase II",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=False,
            time_to_market=2,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        portfolio = Portfolio([Product(product_cfg, model_cfg)], model_cfg)
        engine = MonteCarloEngine(portfolio)

        sims_a = engine.simulate(n_sims=10, launch_delay_sigma=1.0, random_seed=7)
        sims_b = engine.simulate(n_sims=10, launch_delay_sigma=1.0, random_seed=7)
        np.testing.assert_allclose(sims_a.values, sims_b.values)


if __name__ == "__main__":
    unittest.main()
