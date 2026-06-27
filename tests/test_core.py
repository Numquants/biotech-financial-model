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
        self.assertEqual(result.consolidated.loc[2024, "milestones"], 50.0)
        self.assertEqual(result.consolidated.loc[2024, "fcff"], 50.0)
        self.assertEqual(result.rnpv, 50.0)

    def test_stage_transition_probabilities_override_success_prob(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=5)
        product_cfg = ProductConfig(
            name="StageDriven",
            stage="Phase II",
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
        self.assertAlmostEqual(product.effective_success_probability(), 0.25)
        self.assertEqual(product.probability_source(), "stage_transitions")

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
