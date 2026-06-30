"""Core biotech / agro financial modelling primitives."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
import re
from typing import List, Dict, Optional, Callable, Iterable

import numpy as np
import pandas as pd


# =====================================
# 1. Configuration dataclasses (inputs)
# =====================================


@dataclass
class ModelConfig:
    first_year: int = 2024
    n_years: int = 25
    currency: str = "USD"
    tax_rate: float = 0.25
    discount_rate: float = 0.10
    ev_ebitda_multiple: float = 8.0
    working_capital_pct_sales: float = 0.08
    opening_nol_balance: float = 0.0
    sales_ramp_factors: List[float] | None = None
    discount_timing: str = "year_end"
    terminal_method: str = "exit_multiple"
    perpetuity_growth_rate: float = 0.02
    unwind_working_capital: bool = True

    def __post_init__(self) -> None:
        if self.sales_ramp_factors is None:
            self.sales_ramp_factors = [0.20, 0.60, 1.00, 1.00, 1.00]

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.first_year, self.first_year + self.n_years)


@dataclass
class ProductConfig:
    name: str
    stage: str
    success_prob: float
    include_in_consolidation: bool = True

    time_to_market: int = 3
    sales_ramp_length: Optional[int] = None
    sales_ramp_shape: Optional[str] = None
    patent_years: int = 20
    preexisting_market: bool = False

    patent_revenue_target: float = 0.0
    post_patent_revenue_target: float = 0.0
    market_growth_patent: float = 0.005
    market_growth_post: float = 0.0
    patient_population_patent: float = 0.0
    price_per_patient_patent: float = 0.0
    penetration_patent: float = 0.0
    patient_population_post: float = 0.0
    price_per_patient_post: float = 0.0
    penetration_post: float = 0.0

    cogs_patent: float = 0.30
    cogs_post: float = 0.50
    labor_pct: float = 0.12
    overhead_pct: float = 0.08
    material_pct: float = 0.10
    sales_marketing_pct: float = 0.15
    gna_pct: float = 0.10
    royalty_pct: float = 0.0
    commercialization_model: str = "Product Sale"
    launch_delay_sigma_years: float = 0.0

    rd_remaining_pre_launch: float = 0.0
    rd_annual_post_launch: float = 0.0

    capex_remaining_pre_launch: float = 0.0
    capex_annual_post_launch: float = 0.0
    upfront_payment: float = 0.0
    upfront_year_offset: int = 0
    upfront_timing: str = "from_start"

    rd_capitalization_ratio: float = 0.5
    rd_amort_years: int = 10
    capex_dep_years: int = 10

    stage_duration_years: Dict[str, int] = field(default_factory=dict)
    stage_cost_weights: Dict[str, float] = field(default_factory=dict)
    stage_capex_weights: Dict[str, float] = field(default_factory=dict)
    trial_costs_by_phase: Dict[str, float] = field(default_factory=dict)
    stage_transition_probabilities: Dict[str, float] = field(default_factory=dict)
    stage_transition_curve: Dict[str, List[float]] = field(default_factory=dict)
    post_patent_erosion: List[float] = field(default_factory=lambda: [1.0, 0.85, 0.7, 0.55, 0.4])
    milestones: List["Milestone"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage = normalize_stage_label(self.stage)
        self.stage_duration_years = _normalize_stage_value_map(self.stage_duration_years)
        self.stage_cost_weights = _normalize_stage_value_map(self.stage_cost_weights)
        self.stage_capex_weights = _normalize_stage_value_map(self.stage_capex_weights)
        self.trial_costs_by_phase = _normalize_stage_value_map(self.trial_costs_by_phase)
        self.stage_transition_probabilities = _normalize_transition_value_map(
            self.stage_transition_probabilities
        )
        self.stage_transition_curve = _normalize_transition_value_map(self.stage_transition_curve)


@dataclass
class Milestone:
    name: str
    year_offset: int
    amount: float
    probability: float = 1.0
    timing: str = "from_launch"


class Stage(str, Enum):
    DISCOVERY = "Discovery"
    PRECLINICAL = "Preclinical"
    PHASE_I = "Phase I"
    PHASE_II = "Phase II"
    PHASE_III = "Phase III"
    APPROVAL = "Approval"
    COMMERCIAL = "Commercial"


STAGE_SEQUENCE = [stage.value for stage in Stage]
_STAGE_TRANSITIONS = {
    f"{from_stage}->{to_stage}"
    for from_stage, to_stage in zip(STAGE_SEQUENCE[:-1], STAGE_SEQUENCE[1:])
}


def _normalize_stage_token(value: object) -> str:
    return re.sub(r"[\s_-]+", " ", str(value or "").strip()).casefold()


_STAGE_ALIASES = {
    "discovery": Stage.DISCOVERY.value,
    "pre clinical": Stage.PRECLINICAL.value,
    "preclinic": Stage.PRECLINICAL.value,
    "phase 1": Stage.PHASE_I.value,
    "phase 2": Stage.PHASE_II.value,
    "phase 3": Stage.PHASE_III.value,
    "filing": Stage.APPROVAL.value,
    "filed": Stage.APPROVAL.value,
    "registration": Stage.APPROVAL.value,
    "nda filing": Stage.APPROVAL.value,
    "bla filing": Stage.APPROVAL.value,
    "market": Stage.COMMERCIAL.value,
    "marketed": Stage.COMMERCIAL.value,
    "launch": Stage.COMMERCIAL.value,
    "launched": Stage.COMMERCIAL.value,
}
_CANONICAL_STAGES = {_normalize_stage_token(stage): stage for stage in STAGE_SEQUENCE}


def normalize_stage_label(value: str | Stage | None) -> str:
    token = _normalize_stage_token(value)
    if not token:
        return ""
    return _STAGE_ALIASES.get(token, _CANONICAL_STAGES.get(token, str(value).strip()))


def _normalize_transition_key(key: object) -> str:
    raw = str(key or "").strip()
    if "->" not in raw:
        return raw
    left, right = raw.split("->", 1)
    left_stage = normalize_stage_label(left)
    right_stage = normalize_stage_label(right)
    if not left_stage or not right_stage:
        return raw
    return f"{left_stage}->{right_stage}"


def _normalize_stage_value_map(mapping: Optional[Dict[str, object]]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in (mapping or {}).items():
        stage = normalize_stage_label(key)
        if stage:
            normalized[stage] = value
    return normalized


def _normalize_transition_value_map(mapping: Optional[Dict[str, object]]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in (mapping or {}).items():
        transition = _normalize_transition_key(key)
        if transition:
            normalized[transition] = value
    return normalized


def _scale_product_config(
    config: ProductConfig,
    *,
    revenue_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    success_prob_multiplier: float = 1.0,
    launch_delay_years: int = 0,
) -> ProductConfig:
    cfg_dict = asdict(config)
    cfg_dict["patent_revenue_target"] *= revenue_multiplier
    cfg_dict["post_patent_revenue_target"] *= revenue_multiplier
    cfg_dict["cogs_patent"] *= cost_multiplier
    cfg_dict["cogs_post"] *= cost_multiplier
    cfg_dict["labor_pct"] *= cost_multiplier
    cfg_dict["overhead_pct"] *= cost_multiplier
    cfg_dict["material_pct"] *= cost_multiplier
    cfg_dict["sales_marketing_pct"] *= cost_multiplier
    cfg_dict["gna_pct"] *= cost_multiplier
    cfg_dict["royalty_pct"] *= cost_multiplier
    cfg_dict["rd_remaining_pre_launch"] *= cost_multiplier
    cfg_dict["rd_annual_post_launch"] *= cost_multiplier
    cfg_dict["capex_remaining_pre_launch"] *= cost_multiplier
    cfg_dict["capex_annual_post_launch"] *= cost_multiplier
    cfg_dict["trial_costs_by_phase"] = {
        stage: float(amount) * cost_multiplier
        for stage, amount in (config.trial_costs_by_phase or {}).items()
    }
    cfg_dict["success_prob"] = max(0.0, min(1.0, cfg_dict["success_prob"] * success_prob_multiplier))
    cfg_dict["milestones"] = [
        Milestone(**asdict(milestone)) if isinstance(milestone, Milestone) else Milestone(**milestone)
        for milestone in config.milestones
    ]
    if launch_delay_years and not config.preexisting_market:
        cfg_dict["time_to_market"] = max(0, int(config.time_to_market) + int(launch_delay_years))
    return ProductConfig(**cfg_dict)


def _apply_stage_slippage(config: ProductConfig, slippage: Dict[str, int]) -> ProductConfig:
    if not slippage or config.preexisting_market:
        return config
    cfg_dict = asdict(config)
    stage_durations = cfg_dict.get("stage_duration_years") or {}
    updated_durations = dict(stage_durations) if stage_durations else {}
    for stage, years in _normalize_stage_value_map(slippage).items():
        try:
            delay = int(years)
        except (TypeError, ValueError):
            continue
        if delay == 0:
            continue
        if updated_durations:
            updated_durations[stage] = max(0, int(updated_durations.get(stage, 0)) + delay)
        else:
            if stage == config.stage:
                cfg_dict["time_to_market"] = max(0, int(cfg_dict.get("time_to_market", 0)) + delay)
    if updated_durations:
        cfg_dict["stage_duration_years"] = updated_durations
        if config.stage in STAGE_SEQUENCE:
            stage_idx = STAGE_SEQUENCE.index(config.stage)
            total = 0
            for idx in range(stage_idx, len(STAGE_SEQUENCE) - 1):
                from_stage = STAGE_SEQUENCE[idx]
                total += int(updated_durations.get(from_stage, 0))
            cfg_dict["time_to_market"] = max(0, total)
    return ProductConfig(**cfg_dict)


# ===========================
# 2. Core model classes
# ===========================


class Product:
    """Represents a single biotech (or agro) product/asset."""

    def __init__(self, config: ProductConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config

    def _launch_year(self) -> int:
        if self.config.preexisting_market:
            return self.model_config.first_year
        return self.model_config.first_year + max(self.config.time_to_market, 0)

    def _patent_end_year(self) -> int:
        return self._launch_year() + self.config.patent_years - 1

    def probability_source(self) -> str:
        cfg = self.config
        if cfg.stage_transition_curve or cfg.stage_transition_probabilities:
            if cfg.stage in STAGE_SEQUENCE:
                return "stage_transitions"
            return "success_prob_stage_fallback"
        return "success_prob"

    def effective_success_probability(self) -> float:
        return self._stage_success_probability()

    @staticmethod
    def _ramp_factors_array(years_since_launch: np.ndarray, ramp_factors: Iterable[float]) -> np.ndarray:
        ramp_list = list(ramp_factors)
        if not ramp_list:
            ramp = np.ones_like(years_since_launch, dtype=float)
            ramp[years_since_launch < 0] = 0.0
            return ramp
        ramp_values = np.array(ramp_list, dtype=float)
        idx = np.clip(years_since_launch, 0, len(ramp_values) - 1)
        ramp = ramp_values[idx]
        ramp[years_since_launch < 0] = 0.0
        return ramp

    @staticmethod
    def _erosion_factors_array(years_since_patent_end: np.ndarray, erosion_factors: Iterable[float]) -> np.ndarray:
        erosion_list = list(erosion_factors)
        if not erosion_list:
            return np.ones_like(years_since_patent_end, dtype=float)
        erosion_values = np.array(erosion_list, dtype=float)
        idx = np.clip(years_since_patent_end, 0, len(erosion_values) - 1)
        erosion = erosion_values[idx]
        erosion[years_since_patent_end < 0] = 1.0
        return erosion

    @staticmethod
    def _ramp_shape_values(shape: Optional[str], length: int) -> List[float]:
        if length <= 0:
            return []
        normalized = (shape or "Linear").strip().lower()
        if length == 1:
            return [1.0]
        if normalized in {"step", "step-up"}:
            return [0.0] + [1.0] * (length - 1)
        if normalized in {"s-curve", "s curve", "sigmoid"}:
            x = np.linspace(-2.5, 2.5, length)
            curve = 1.0 / (1.0 + np.exp(-x))
            curve = (curve - curve.min()) / (curve.max() - curve.min())
            return curve.tolist()
        return np.linspace(0.2, 1.0, length).tolist()

    def _stage_success_probability(self) -> float:
        cfg = self.config
        if not cfg.stage_transition_probabilities and not cfg.stage_transition_curve:
            return max(0.0, min(1.0, cfg.success_prob))
        if cfg.stage not in STAGE_SEQUENCE:
            return max(0.0, min(1.0, cfg.success_prob))
        stage_idx = STAGE_SEQUENCE.index(cfg.stage)
        if cfg.stage in {"Approval", "Commercial"}:
            return 1.0
        transitions = []
        for idx in range(stage_idx, len(STAGE_SEQUENCE) - 1):
            from_stage = STAGE_SEQUENCE[idx]
            to_stage = STAGE_SEQUENCE[idx + 1]
            key = f"{from_stage}->{to_stage}"
            if cfg.stage_transition_curve and key in cfg.stage_transition_curve:
                curve = cfg.stage_transition_curve.get(key, [])
                transitions.extend(curve if curve else [1.0])
            else:
                transitions.append(cfg.stage_transition_probabilities.get(key, 1.0))
        prob = float(np.prod(transitions)) if transitions else 1.0
        return max(0.0, min(1.0, prob))

    def _resolved_revenue_targets(self) -> tuple[float, float]:
        cfg = self.config
        patent_target = float(cfg.patent_revenue_target or 0.0)
        post_target = float(cfg.post_patent_revenue_target or 0.0)

        if (
            cfg.patient_population_patent > 0
            and cfg.price_per_patient_patent > 0
            and cfg.penetration_patent > 0
        ):
            patent_target = (
                float(cfg.patient_population_patent)
                * float(cfg.price_per_patient_patent)
                * float(cfg.penetration_patent)
            )

        if (
            cfg.patient_population_post > 0
            and cfg.price_per_patient_post > 0
            and cfg.penetration_post > 0
        ):
            post_target = (
                float(cfg.patient_population_post)
                * float(cfg.price_per_patient_post)
                * float(cfg.penetration_post)
            )

        return patent_target, post_target

    def _success_prob_schedule(self) -> pd.Series:
        years = self.model_config.years
        cfg = self.config
        cumulative_prob = self._stage_success_probability()
        if cfg.preexisting_market or cfg.time_to_market <= 0:
            return pd.Series(cumulative_prob, index=years)
        if cfg.stage_transition_curve or cfg.stage_duration_years:
            annual_probs: List[float] = []
            if cfg.stage in STAGE_SEQUENCE:
                stage_idx = STAGE_SEQUENCE.index(cfg.stage)
                for idx in range(stage_idx, len(STAGE_SEQUENCE) - 1):
                    from_stage = STAGE_SEQUENCE[idx]
                    to_stage = STAGE_SEQUENCE[idx + 1]
                    key = f"{from_stage}->{to_stage}"
                    duration = int(cfg.stage_duration_years.get(from_stage, 0))
                    if duration <= 0:
                        continue
                    curve = cfg.stage_transition_curve.get(key)
                    if curve:
                        curve_list = [float(value) for value in curve]
                        if len(curve_list) < duration:
                            curve_list = curve_list + [curve_list[-1]] * (duration - len(curve_list))
                        else:
                            curve_list = curve_list[:duration]
                        annual_probs.extend(curve_list)
                    else:
                        annual_prob = cfg.stage_transition_probabilities.get(key, 1.0)
                        annual_probs.extend([annual_prob] * duration)
            if annual_probs:
                schedule = np.ones(len(years), dtype=float)
                cumulative = 1.0
                max_years = min(len(annual_probs), len(years))
                for idx in range(max_years):
                    cumulative *= annual_probs[idx]
                    schedule[idx] = cumulative
                schedule[max_years:] = cumulative
                return pd.Series(schedule, index=years)
        pre_years = max(1, int(cfg.time_to_market))
        schedule = np.ones(len(years), dtype=float)
        ramp = np.linspace(1.0, cumulative_prob, num=pre_years + 1)
        end_idx = min(len(schedule), len(ramp))
        schedule[:end_idx] = ramp[:end_idx]
        if end_idx < len(schedule):
            schedule[end_idx:] = cumulative_prob
        return pd.Series(schedule, index=years)

    @staticmethod
    def _growth_years(
        years: np.ndarray,
        years_since_launch: np.ndarray,
        in_patent: np.ndarray,
        ramp_length: int,
        patent_end: int,
    ) -> np.ndarray:
        growth_years = np.zeros_like(years_since_launch, dtype=float)
        if ramp_length > 0:
            growth_years[in_patent] = np.maximum(0, years_since_launch[in_patent] - ramp_length)
        else:
            growth_years[in_patent] = np.maximum(0, years_since_launch[in_patent])
        growth_years[~in_patent] = np.maximum(0, years[~in_patent] - (patent_end + 1))
        return growth_years

    @staticmethod
    def _rolling_amortization(additions: pd.Series, life: int) -> pd.Series:
        if life is None or not np.isfinite(life):
            return pd.Series(0.0, index=additions.index)
        life = int(life)
        if life <= 0:
            return pd.Series(0.0, index=additions.index)
        weights = np.ones(life) / life
        amort_values = np.convolve(additions.values, weights, mode="full")[: len(additions)]
        return pd.Series(amort_values, index=additions.index)

    @staticmethod
    def _spread_prelaunch_cashflow(total: float, years: int, index: np.ndarray) -> pd.Series:
        values = np.zeros(len(index))
        if total <= 0 or years <= 0:
            return pd.Series(values, index=index)
        annual = total / years
        values[:years] = -annual
        return pd.Series(values, index=index)

    def _stage_costed_prelaunch_cashflow(
        self,
        total: float,
        index: np.ndarray,
        stage_weights: Dict[str, float],
    ) -> pd.Series:
        cfg = self.config
        values = np.zeros(len(index))
        if total <= 0 or cfg.preexisting_market:
            return pd.Series(values, index=index)
        if not cfg.stage_duration_years:
            pre_years = max(1, int(cfg.time_to_market))
            return self._spread_prelaunch_cashflow(total, pre_years, index)
        if cfg.stage not in STAGE_SEQUENCE:
            pre_years = max(1, int(cfg.time_to_market))
            return self._spread_prelaunch_cashflow(total, pre_years, index)
        stage_idx = STAGE_SEQUENCE.index(cfg.stage)
        stages = STAGE_SEQUENCE[stage_idx : -1]
        total_years = sum(int(cfg.stage_duration_years.get(stage, 0)) for stage in stages)
        if total_years <= 0:
            pre_years = max(1, int(cfg.time_to_market))
            return self._spread_prelaunch_cashflow(total, pre_years, index)
        weights = {stage: float(stage_weights.get(stage, 0.0)) for stage in stages}
        weight_total = sum(weight for stage, weight in weights.items() if cfg.stage_duration_years.get(stage, 0) > 0)
        per_year: List[float] = []
        if weight_total <= 0:
            per_year = [total / total_years] * total_years
        else:
            for stage in stages:
                duration = int(cfg.stage_duration_years.get(stage, 0))
                if duration <= 0:
                    continue
                stage_weight = weights.get(stage, 0.0) / weight_total
                stage_total = total * stage_weight
                per_year.extend([stage_total / duration] * duration)
        pre_years = min(len(per_year), len(index))
        values[:pre_years] = -np.array(per_year[:pre_years])
        return pd.Series(values, index=index)

    def _phase_explicit_prelaunch_cashflow(
        self,
        phase_amounts: Dict[str, float],
        index: np.ndarray,
    ) -> pd.Series:
        cfg = self.config
        values = np.zeros(len(index))
        if not phase_amounts or cfg.preexisting_market or cfg.stage not in STAGE_SEQUENCE:
            return pd.Series(values, index=index)

        stage_idx = STAGE_SEQUENCE.index(cfg.stage)
        phases = STAGE_SEQUENCE[stage_idx:-1]
        offset = 0
        for phase in phases:
            duration = max(0, int(cfg.stage_duration_years.get(phase, 0)))
            if duration <= 0:
                continue
            phase_total = max(0.0, float(phase_amounts.get(phase, 0.0)))
            annual = phase_total / duration if duration else 0.0
            next_offset = min(offset + duration, len(index))
            values[offset:next_offset] = -annual
            offset = next_offset
            if offset >= len(index):
                break
        return pd.Series(values, index=index)

    def _normalized_milestones(self) -> List[Milestone]:
        normalized: List[Milestone] = []
        for milestone in self.config.milestones:
            if isinstance(milestone, Milestone):
                normalized.append(milestone)
                continue
            try:
                normalized.append(Milestone(**milestone))
            except (TypeError, ValueError, KeyError):
                continue
        if float(self.config.upfront_payment or 0.0) != 0.0:
            normalized.append(
                Milestone(
                    name="Upfront payment",
                    year_offset=int(self.config.upfront_year_offset),
                    amount=float(self.config.upfront_payment),
                    probability=1.0,
                    timing=self.config.upfront_timing or "from_start",
                )
            )
        return normalized

    def build_revenue_series(self) -> pd.Series:
        years = self.model_config.years
        cfg = self.config
        ramp_factors = list(self.model_config.sales_ramp_factors or [])
        if cfg.sales_ramp_shape:
            ramp_len = int(cfg.sales_ramp_length or len(ramp_factors) or 1)
            ramp_factors = self._ramp_shape_values(cfg.sales_ramp_shape, ramp_len)
        elif cfg.sales_ramp_length is not None:
            ramp_len = int(cfg.sales_ramp_length)
            if ramp_len <= 0:
                ramp_factors = []
            elif not ramp_factors:
                ramp_factors = [1.0] * ramp_len
            elif ramp_len <= len(ramp_factors):
                ramp_factors = ramp_factors[:ramp_len]
            else:
                ramp_factors = ramp_factors + [ramp_factors[-1]] * (ramp_len - len(ramp_factors))
        revenue = pd.Series(0.0, index=years, name=f"{cfg.name}_revenue")
        if not cfg.include_in_consolidation:
            return revenue

        years_arr = np.asarray(years, dtype=int)
        launch_year = self._launch_year()
        patent_end = self._patent_end_year()
        years_since_launch = years_arr - launch_year
        in_patent = years_arr <= patent_end
        patent_target, post_target = self._resolved_revenue_targets()

        ramp = self._ramp_factors_array(years_since_launch, ramp_factors)
        base_target = np.where(in_patent, patent_target, post_target)
        growth_rate = np.where(in_patent, cfg.market_growth_patent, cfg.market_growth_post)
        growth_years = self._growth_years(
            years_arr,
            years_since_launch,
            in_patent,
            len(ramp_factors),
            patent_end,
        )
        target_with_growth = base_target * np.power(1.0 + growth_rate, growth_years)
        years_since_patent_end = years_arr - (patent_end + 1)
        erosion = self._erosion_factors_array(years_since_patent_end, cfg.post_patent_erosion)
        revenue_data = ramp * target_with_growth * erosion
        return pd.Series(revenue_data, index=years, name=revenue.name)

    def build_cashflow_table(self) -> pd.DataFrame:
        years = self.model_config.years
        cfg = self.config
        df = pd.DataFrame(index=years)
        df["revenue"] = self.build_revenue_series()

        patent_end = self._patent_end_year()
        in_patent = df.index.values <= patent_end
        cogs_pct = np.where(in_patent, cfg.cogs_patent, cfg.cogs_post)
        df["cogs"] = -df["revenue"].values * cogs_pct

        pct_columns = {
            "labor": cfg.labor_pct,
            "overhead": cfg.overhead_pct,
            "materials": cfg.material_pct,
            "sales_marketing": cfg.sales_marketing_pct,
            "gna": cfg.gna_pct,
            "royalty": cfg.royalty_pct,
        }
        for col, pct in pct_columns.items():
            df[col] = -pct * df["revenue"]

        rd_cash = pd.Series(0.0, index=years)
        if cfg.rd_remaining_pre_launch > 0 and not cfg.preexisting_market:
            if cfg.trial_costs_by_phase:
                rd_cash += self._phase_explicit_prelaunch_cashflow(
                    cfg.trial_costs_by_phase,
                    years,
                )
            else:
                rd_cash += self._stage_costed_prelaunch_cashflow(
                    cfg.rd_remaining_pre_launch,
                    years,
                    cfg.stage_cost_weights,
                )

        launch_year = self._launch_year()
        rd_cash.loc[years >= launch_year] -= cfg.rd_annual_post_launch
        df["rd_cash"] = rd_cash

        rd_cap_add = rd_cash * cfg.rd_capitalization_ratio
        rd_expensed_current = rd_cash * (1 - cfg.rd_capitalization_ratio)
        rd_amort = self._rolling_amortization(rd_cap_add, cfg.rd_amort_years)
        df["rd_cap_add"] = rd_cap_add
        df["rd_amort"] = rd_amort
        df["rd_expense_pnl"] = rd_expensed_current + rd_amort

        capex_cash = pd.Series(0.0, index=years)
        if cfg.capex_remaining_pre_launch > 0 and not cfg.preexisting_market:
            capex_cash += self._stage_costed_prelaunch_cashflow(
                cfg.capex_remaining_pre_launch,
                years,
                cfg.stage_capex_weights,
            )

        capex_cash.loc[years >= launch_year] -= cfg.capex_annual_post_launch
        df["capex_cash"] = capex_cash

        depreciation = self._rolling_amortization(capex_cash, cfg.capex_dep_years)
        df["depreciation"] = depreciation

        df["ebit"] = (
            df["revenue"]
            + df["cogs"]
            + df["labor"]
            + df["overhead"]
            + df["materials"]
            + df["sales_marketing"]
            + df["gna"]
            + df["royalty"]
            + df["rd_expense_pnl"]
        )
        df["milestones"] = 0.0
        for milestone in self._normalized_milestones():
            if milestone.timing == "from_launch":
                milestone_year = self._launch_year() + milestone.year_offset
            elif milestone.timing == "from_start":
                milestone_year = self.model_config.first_year + milestone.year_offset
            else:
                milestone_year = self.model_config.first_year + milestone.year_offset
            if milestone_year in df.index:
                weighted_amount = float(milestone.amount) * float(milestone.probability)
                df.loc[milestone_year, "milestones"] += weighted_amount
        df["ebit"] = df["ebit"] + df["milestones"]
        df["da"] = -(df["rd_amort"] + df["depreciation"])
        df["ebitda"] = df["ebit"] + df["da"]

        tax_rate = self.model_config.tax_rate
        df["tax"] = 0.0
        positive_ebit = df["ebit"] > 0
        df.loc[positive_ebit, "tax"] = -tax_rate * df.loc[positive_ebit, "ebit"]
        df["nopat"] = df["ebit"] + df["tax"]

        df["fcff"] = df["nopat"] + df["da"] + df["capex_cash"] + df["rd_cap_add"]
        return df

    def build_probability_weighted_table(self) -> pd.DataFrame:
        df = self.build_cashflow_table().copy()
        prob_schedule = self._success_prob_schedule()
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].multiply(prob_schedule, axis=0)
        return df


class Portfolio:
    """A collection of products that can be valued together."""

    def __init__(self, products: List[Product], model_config: ModelConfig):
        self.products = products
        self.model_config = model_config

    def _apply_consolidated_tax(self, cons_df: pd.DataFrame) -> pd.DataFrame:
        nol_balance = max(0.0, float(self.model_config.opening_nol_balance or 0.0))
        tax_rate = float(self.model_config.tax_rate)
        taxes: List[float] = []
        taxable_after_nol: List[float] = []
        nol_used: List[float] = []
        nol_balances: List[float] = []

        for ebit in cons_df["ebit"].tolist():
            ebit_val = float(ebit)
            applied_nol = 0.0
            taxable = max(0.0, ebit_val)
            if taxable > 0 and nol_balance > 0:
                applied_nol = min(taxable, nol_balance)
                taxable -= applied_nol
                nol_balance -= applied_nol
            if ebit_val < 0:
                nol_balance += -ebit_val

            taxes.append(-tax_rate * taxable)
            taxable_after_nol.append(taxable)
            nol_used.append(applied_nol)
            nol_balances.append(nol_balance)

        cons_df["taxable_income_after_nol"] = taxable_after_nol
        cons_df["nol_utilized"] = nol_used
        cons_df["nol_balance"] = nol_balances
        cons_df["tax"] = taxes
        cons_df["nopat"] = cons_df["ebit"] + cons_df["tax"]
        cons_df["fcff"] = cons_df["nopat"] + cons_df["da"] + cons_df["capex_cash"] + cons_df["rd_cap_add"]
        return cons_df

    def consolidated_table(self) -> Dict[str, pd.DataFrame | pd.Series]:
        years = self.model_config.years
        base_cols = [
            "revenue",
            "cogs",
            "labor",
            "overhead",
            "materials",
            "sales_marketing",
            "gna",
            "royalty",
            "milestones",
            "rd_cash",
            "rd_cap_add",
            "rd_amort",
            "rd_expense_pnl",
            "capex_cash",
            "depreciation",
            "ebit",
            "da",
            "ebitda",
            "tax",
            "nopat",
            "fcff",
        ]
        cons_df = pd.DataFrame(0.0, index=years, columns=base_cols)

        per_product: Dict[str, pd.DataFrame] = {}
        per_product_prob: Dict[str, pd.DataFrame] = {}

        for prod in self.products:
            cfg = prod.config
            if not cfg.include_in_consolidation:
                continue
            df = prod.build_cashflow_table()
            per_product[cfg.name] = df
            wdf = prod.build_probability_weighted_table()
            per_product_prob[cfg.name] = wdf
            cons_df = cons_df.add(wdf[base_cols], fill_value=0.0)

        wc = self.model_config.working_capital_pct_sales * cons_df["revenue"]
        wc_diff = wc.diff().fillna(wc)
        cons_df["working_capital_balance"] = wc
        cons_df["delta_wc"] = -wc_diff
        cons_df = self._apply_consolidated_tax(cons_df)
        cons_df["fcff_after_wc"] = cons_df["fcff"] + cons_df["delta_wc"]

        return {
            "per_product": per_product,
            "per_product_prob": per_product_prob,
            "consolidated": cons_df,
        }


# ======================
# 3. Valuation engine
# ======================


@dataclass
class ValuationResult:
    portfolio: Portfolio
    rnpv: float
    dcf_table: pd.DataFrame
    consolidated: pd.DataFrame
    per_product: Dict[str, pd.DataFrame]
    per_product_prob: Dict[str, pd.DataFrame]

    @property
    def enterprise_value(self) -> float:
        return self.rnpv


class ValuationEngine:
    """Runs DCF valuation (rNPV, terminal value) on a Portfolio."""

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.model_config = portfolio.model_config

    def _discount_periods(self, n_periods: int) -> np.ndarray:
        timing = (self.model_config.discount_timing or "year_end").strip().lower()
        if timing == "year_0":
            return np.arange(0, n_periods, dtype=float)
        if timing == "mid_year":
            return np.arange(0, n_periods, dtype=float) + 0.5
        return np.arange(1, n_periods + 1, dtype=float)

    def _discounted_cash_flows(self, fcff: pd.Series) -> pd.DataFrame:
        years = fcff.index.values
        t = self._discount_periods(len(years))
        df = pd.DataFrame(index=years)
        df["t"] = t
        df["fcff"] = fcff.values
        df["discount_factor"] = 1.0 / ((1 + self.model_config.discount_rate) ** t)
        df["discounted_fcff"] = df["fcff"] * df["discount_factor"]
        return df

    def _add_terminal_value(self, dcf_df: pd.DataFrame, cons_df: pd.DataFrame) -> float:
        last_year = cons_df.index[-1]
        terminal_method = (self.model_config.terminal_method or "exit_multiple").strip().lower()
        if terminal_method == "perpetuity_growth":
            growth = float(self.model_config.perpetuity_growth_rate)
            rate = float(self.model_config.discount_rate)
            if rate <= growth:
                raise ValueError("discount_rate must exceed perpetuity_growth_rate for perpetuity terminal value")
            terminal_base = float(cons_df.loc[last_year, "fcff_after_wc"])
            terminal_ev = (terminal_base * (1.0 + growth)) / (rate - growth)
        else:
            last_ebitda = cons_df.loc[last_year, "ebitda"]
            multiple = self.model_config.ev_ebitda_multiple
            terminal_ebitda = max(0.0, float(last_ebitda))
            terminal_ev = multiple * terminal_ebitda

        t_last = dcf_df.loc[last_year, "t"]
        dcf_df.loc[last_year, "terminal_value"] = terminal_ev
        dcf_df.loc[last_year, "discounted_terminal_value"] = terminal_ev / (
            (1 + self.model_config.discount_rate) ** t_last
        )
        wc_recovery = 0.0
        if self.model_config.unwind_working_capital:
            wc_recovery = max(0.0, float(cons_df.get("working_capital_balance", pd.Series(0.0, index=cons_df.index)).iloc[-1]))
        dcf_df.loc[last_year, "working_capital_recovery"] = wc_recovery
        dcf_df.loc[last_year, "discounted_working_capital_recovery"] = wc_recovery / (
            (1 + self.model_config.discount_rate) ** t_last
        )
        rnpv = (
            dcf_df["discounted_fcff"].sum()
            + dcf_df["discounted_terminal_value"].sum()
            + dcf_df["discounted_working_capital_recovery"].sum()
        )
        return rnpv

    def run(self) -> ValuationResult:
        agg = self.portfolio.consolidated_table()
        cons = agg["consolidated"]
        dcf_df = self._discounted_cash_flows(cons["fcff_after_wc"])
        rnpv = self._add_terminal_value(dcf_df, cons)
        return ValuationResult(
            portfolio=self.portfolio,
            rnpv=rnpv,
            dcf_table=dcf_df,
            consolidated=cons,
            per_product=agg["per_product"],
            per_product_prob=agg["per_product_prob"],
        )


# ======================
# 4. VC-style valuation
# ======================


@dataclass
class VCInputs:
    exit_year: int
    target_irr: float
    investor_ownership_at_exit: float
    new_money: float


class VCValuator:
    def __init__(self, valuation_result: ValuationResult):
        self.result = valuation_result
        self.model_config = valuation_result.portfolio.model_config

    def compute_exit_ev(self, exit_year: int, multiple: Optional[float] = None) -> float:
        if multiple is None:
            multiple = self.model_config.ev_ebitda_multiple
        cons = self.result.consolidated
        if exit_year not in cons.index:
            raise ValueError("Exit year not in consolidated index")
        exit_ebitda = cons.loc[exit_year, "ebitda"]
        return float(exit_ebitda * multiple)

    def vc_method(self, vc_inputs: VCInputs, exit_multiple: Optional[float] = None) -> Dict[str, float]:
        exit_ev = self.compute_exit_ev(vc_inputs.exit_year, exit_multiple)
        years_to_exit = vc_inputs.exit_year - self.model_config.first_year
        if years_to_exit <= 0:
            raise ValueError("Exit year must be after first model year")

        investor_exit_value = exit_ev * vc_inputs.investor_ownership_at_exit
        investor_pv_required = investor_exit_value / ((1 + vc_inputs.target_irr) ** years_to_exit)
        implied_post_money = investor_pv_required / vc_inputs.investor_ownership_at_exit
        implied_pre_money = implied_post_money - vc_inputs.new_money
        investor_irr_actual = (investor_exit_value / vc_inputs.new_money) ** (1 / years_to_exit) - 1

        return {
            "exit_enterprise_value": exit_ev,
            "investor_exit_value": investor_exit_value,
            "investor_pv_required": investor_pv_required,
            "implied_post_money": implied_post_money,
            "implied_pre_money": implied_pre_money,
            "investor_irr_if_pay_new_money": investor_irr_actual,
        }


# ============================
# 5. Scenario & stress testing
# ============================


@dataclass
class Scenario:
    name: str
    revenue_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    discount_rate_shift: float = 0.0
    success_prob_multiplier: float = 1.0
    launch_delay_years: int = 0
    stage_slippage_years: Dict[str, int] = field(default_factory=dict)


class ScenarioEngine:
    def __init__(self, base_portfolio: Portfolio):
        self.base_portfolio = base_portfolio
        self.base_model_config = base_portfolio.model_config

    def _apply_scenario(self, scenario: Scenario) -> Portfolio:
        new_model_cfg = ModelConfig(**asdict(self.base_model_config))
        new_model_cfg.discount_rate += scenario.discount_rate_shift

        new_products: List[Product] = []
        for prod in self.base_portfolio.products:
            new_cfg = _scale_product_config(
                prod.config,
                revenue_multiplier=scenario.revenue_multiplier,
                cost_multiplier=scenario.cost_multiplier,
                success_prob_multiplier=scenario.success_prob_multiplier,
                launch_delay_years=scenario.launch_delay_years,
            )
            new_cfg = _apply_stage_slippage(new_cfg, scenario.stage_slippage_years)
            new_products.append(Product(new_cfg, new_model_cfg))

        return Portfolio(new_products, new_model_cfg)

    def run_scenarios(self, scenarios: List[Scenario], ebitda_year_offset: int = 0) -> pd.DataFrame:
        rows = []

        base_val = ValuationEngine(self.base_portfolio).run()
        base_cons = base_val.consolidated
        base_year = self.base_model_config.first_year + ebitda_year_offset
        rows.append(
            {
                "scenario": "Base",
                "discount_rate": self.base_model_config.discount_rate,
                "rnpv": base_val.rnpv,
                "ebitda_year": base_year,
                "ebitda_value": float(base_cons.loc[base_year, "ebitda"]),
            }
        )

        for sc in scenarios:
            port_sc = self._apply_scenario(sc)
            val = ValuationEngine(port_sc).run()
            cons = val.consolidated
            year = port_sc.model_config.first_year + ebitda_year_offset
            rows.append(
                {
                    "scenario": sc.name,
                    "discount_rate": port_sc.model_config.discount_rate,
                    "rnpv": val.rnpv,
                    "ebitda_year": year,
                    "ebitda_value": float(cons.loc[year, "ebitda"]),
                }
            )

        return pd.DataFrame(rows)


# ==========================
# 6. Monte Carlo & risk
# ==========================


class MonteCarloEngine:
    def __init__(self, base_portfolio: Portfolio):
        self.base_portfolio = base_portfolio

    def simulate(
        self,
        n_sims: int = 1000,
        revenue_sigma: float = 0.1,
        cost_sigma: float = 0.05,
        revenue_dist: str = "normal",
        cost_dist: str = "normal",
        revenue_min: float = 0.8,
        revenue_max: float = 1.2,
        cost_min: float = 0.8,
        cost_max: float = 1.2,
        revenue_cost_correlation: float = 0.3,
        launch_delay_sigma: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> pd.Series:
        rng = np.random.default_rng(random_seed)
        vals = []

        def _sample_scale(dist: str, z: float, sigma: float, min_val: float, max_val: float) -> float:
            dist = dist.lower()
            if dist == "lognormal":
                mu = -0.5 * sigma**2
                return float(np.exp(mu + sigma * z))
            if dist == "uniform":
                low = min(min_val, max_val)
                high = max(min_val, max_val)
                u = float(0.5 * (1 + np.math.erf(z / np.sqrt(2))))
                return float(low + (high - low) * u)
            return float(1.0 + sigma * z)

        for _ in range(n_sims):
            model_cfg = self.base_portfolio.model_config
            new_products: List[Product] = []
            corr = float(np.clip(revenue_cost_correlation, -0.99, 0.99))
            cov = np.array([[1.0, corr], [corr, 1.0]])
            z = rng.multivariate_normal([0.0, 0.0], cov)
            rev_scale = _sample_scale(revenue_dist, z[0], revenue_sigma, revenue_min, revenue_max)
            cogs_scale = _sample_scale(cost_dist, z[1], cost_sigma, cost_min, cost_max)

            for prod in self.base_portfolio.products:
                delay_sigma = max(
                    float(launch_delay_sigma),
                    float(getattr(prod.config, "launch_delay_sigma_years", 0.0) or 0.0),
                )
                sampled_delay = int(np.rint(rng.normal(0.0, delay_sigma))) if delay_sigma > 0 else 0
                new_cfg = _scale_product_config(
                    prod.config,
                    revenue_multiplier=rev_scale,
                    cost_multiplier=cogs_scale,
                    launch_delay_years=sampled_delay,
                )
                new_products.append(Product(new_cfg, model_cfg))

            sim_portfolio = Portfolio(new_products, model_cfg)
            val = ValuationEngine(sim_portfolio).run()
            vals.append(val.rnpv)

        return pd.Series(vals, name="rnpv_sim")

    @staticmethod
    def value_at_risk(simulated_rnpv: pd.Series, alpha: float = 0.95) -> float:
        return float(simulated_rnpv.quantile(1 - alpha))

    @staticmethod
    def conditional_value_at_risk(simulated_rnpv: pd.Series, alpha: float = 0.95) -> float:
        var = simulated_rnpv.quantile(1 - alpha)
        tail = simulated_rnpv[simulated_rnpv <= var]
        if len(tail) == 0:
            return float(var)
        return float(tail.mean())


# ======================================
# 8. ForecastEngine: ARIMA / Prophet / LSTM
# ======================================


class ForecastEngine:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def forecast_arima(
        self,
        series: pd.Series,
        order: tuple[int, int, int] = (1, 1, 1),
        steps: Optional[int] = None,
    ) -> pd.Series:
        from statsmodels.tsa.arima.model import ARIMA

        if steps is None:
            steps = self.model_config.n_years

        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        forecast.name = f"{series.name}_arima_forecast"
        return forecast

    def forecast_prophet(
        self,
        df: pd.DataFrame,
        periods: Optional[int] = None,
        freq: str = "Y",
    ) -> pd.DataFrame:
        from prophet import Prophet

        if periods is None:
            periods = self.model_config.n_years

        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)
        return forecast

    def forecast_lstm(
        self,
        series: pd.Series,
        lookback: int = 12,
        steps_ahead: Optional[int] = None,
        epochs: int = 50,
        batch_size: int = 16,
    ) -> np.ndarray:
        import numpy as np
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        if steps_ahead is None:
            steps_ahead = self.model_config.n_years

        values = series.values.astype("float32")
        X, y = [], []
        for i in range(len(values) - lookback):
            X.append(values[i : i + lookback])
            y.append(values[i + lookback])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(32, input_shape=(lookback, 1)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        history = values[-lookback:].copy()
        forecasts = []
        for _ in range(steps_ahead):
            x_input = history[-lookback:].reshape((1, lookback, 1))
            yhat = model.predict(x_input, verbose=0)[0, 0]
            forecasts.append(yhat)
            history = np.append(history, yhat)

        return np.array(forecasts)

    @staticmethod
    def _implied_cagr_from_forecast(base_value: float, forecast_values: np.ndarray) -> float:
        if base_value <= 0:
            return 0.0
        horizon = len(forecast_values)
        if horizon <= 0:
            return 0.0
        terminal_value = float(forecast_values[-1])
        if terminal_value <= 0:
            return 0.0
        return (terminal_value / base_value) ** (1 / horizon) - 1

    def apply_price_forecast_to_products(
        self,
        products: List[Product],
        price_forecast: pd.Series,
        base_price: float,
        mode: str = "growth",
        growth_scale: float = 1.0,
        revenue_scale_max: float = 2.0,
    ) -> List[Product]:
        forecast_values = price_forecast.values
        out: List[Product] = []

        if mode == "growth":
            implied_cagr = self._implied_cagr_from_forecast(base_price, forecast_values)
            adj = growth_scale * implied_cagr

            for prod in products:
                cfg_dict = asdict(prod.config)
                cfg_dict["market_growth_patent"] += adj
                cfg_dict["market_growth_post"] += adj
                new_cfg = ProductConfig(**cfg_dict)
                out.append(Product(new_cfg, prod.model_config))

        elif mode == "revenue_scale":
            ratio = float(np.mean(forecast_values)) / base_price if base_price > 0 else 1.0
            ratio = min(max(ratio, 0.0), revenue_scale_max)

            for prod in products:
                cfg_dict = asdict(prod.config)
                cfg_dict["patent_revenue_target"] *= ratio
                cfg_dict["post_patent_revenue_target"] *= ratio
                new_cfg = ProductConfig(**cfg_dict)
                out.append(Product(new_cfg, prod.model_config))

        else:
            raise ValueError("mode must be 'growth' or 'revenue_scale'")

        return out


# ===========================================================
# 9. Forecast → Scenario bridge (Prophet-driven stress tests)
# ===========================================================


class ForecastScenarioBridge:
    def __init__(self, base_portfolio: Portfolio, forecast_engine: ForecastEngine):
        self.base_portfolio = base_portfolio
        self.forecast_engine = forecast_engine
        self.model_config = base_portfolio.model_config

    @staticmethod
    def _avg_ratio(series: pd.Series, base_value: float, cap: float = 3.0) -> float:
        if base_value <= 0:
            return 1.0
        ratio = float(series.mean() / base_value)
        return max(0.0, min(ratio, cap))

    def build_price_scenarios_from_prophet(
        self,
        hist_df: pd.DataFrame,
        periods: Optional[int] = None,
        freq: str = "Y",
        pessimistic_discount_uplift: float = 0.02,
        optimistic_discount_reduction: float = 0.01,
        cost_sensitivity: float = 0.05,
        ebitda_year_offset: int = 0,
    ) -> pd.DataFrame:
        forecast = self.forecast_engine.forecast_prophet(hist_df, periods=periods, freq=freq)
        last_hist_date = hist_df["ds"].max()
        future_fc = forecast[forecast["ds"] > last_hist_date].copy()
        n = self.model_config.n_years
        future_fc = future_fc.head(n)

        base_series = future_fc["yhat"]
        pess_series = future_fc["yhat_lower"]
        opt_series = future_fc["yhat_upper"]

        base_price = float(hist_df["y"].iloc[-1])
        base_ratio = self._avg_ratio(base_series, base_price)
        pess_ratio = self._avg_ratio(pess_series, base_price)
        opt_ratio = self._avg_ratio(opt_series, base_price)

        scenarios: List[Scenario] = []
        scenarios.append(
            Scenario(
                name="Prophet Base",
                revenue_multiplier=base_ratio,
                cost_multiplier=1.0,
                discount_rate_shift=0.0,
                success_prob_multiplier=1.0,
            )
        )
        scenarios.append(
            Scenario(
                name="Prophet Pessimistic",
                revenue_multiplier=pess_ratio,
                cost_multiplier=1.0 + cost_sensitivity,
                discount_rate_shift=pessimistic_discount_uplift,
                success_prob_multiplier=0.95,
            )
        )
        scenarios.append(
            Scenario(
                name="Prophet Optimistic",
                revenue_multiplier=opt_ratio,
                cost_multiplier=max(0.0, 1.0 - cost_sensitivity),
                discount_rate_shift=-optimistic_discount_reduction,
                success_prob_multiplier=1.05,
            )
        )

        scen_engine = ScenarioEngine(self.base_portfolio)
        scen_results = scen_engine.run_scenarios(
            scenarios=scenarios,
            ebitda_year_offset=ebitda_year_offset,
        )
        return scen_results


__all__ = [
    "ModelConfig",
    "ProductConfig",
    "Milestone",
    "Stage",
    "STAGE_SEQUENCE",
    "normalize_stage_label",
    "Product",
    "Portfolio",
    "ValuationEngine",
    "ValuationResult",
    "VCInputs",
    "VCValuator",
    "Scenario",
    "ScenarioEngine",
    "MonteCarloEngine",
    "ForecastEngine",
    "ForecastScenarioBridge",
    "validate_model_config",
    "validate_product_config",
    "validate_portfolio",
]


def validate_model_config(config: ModelConfig) -> List[str]:
    issues: List[str] = []
    if config.n_years <= 0:
        issues.append("ModelConfig: n_years must be positive.")
    if not (0.0 <= config.tax_rate <= 0.6):
        issues.append("ModelConfig: tax_rate must be between 0 and 0.6.")
    if not (0.0 <= config.discount_rate <= 0.6):
        issues.append("ModelConfig: discount_rate must be between 0 and 0.6.")
    if not (0.0 <= config.working_capital_pct_sales <= 1.0):
        issues.append("ModelConfig: working_capital_pct_sales must be between 0 and 1.")
    if not (0.0 <= config.ev_ebitda_multiple <= 30.0):
        issues.append("ModelConfig: ev_ebitda_multiple must be between 0 and 30.")
    if config.opening_nol_balance < 0:
        issues.append("ModelConfig: opening_nol_balance cannot be negative.")
    if config.terminal_method not in {"exit_multiple", "perpetuity_growth"}:
        issues.append("ModelConfig: terminal_method must be exit_multiple or perpetuity_growth.")
    if config.terminal_method == "perpetuity_growth":
        if not (-0.05 <= config.perpetuity_growth_rate <= 0.06):
            issues.append("ModelConfig: perpetuity_growth_rate must be between -5% and 6%.")
        if config.perpetuity_growth_rate >= config.discount_rate:
            issues.append("ModelConfig: perpetuity_growth_rate must be lower than discount_rate.")
    if config.discount_timing not in {"year_end", "mid_year", "year_0"}:
        issues.append("ModelConfig: discount_timing must be year_end, mid_year, or year_0.")
    for factor in config.sales_ramp_factors or []:
        if factor < 0 or factor > 2.0:
            issues.append("ModelConfig: sales_ramp_factors must stay between 0 and 2.")
            break
    return issues


def _validate_patient_target_block(
    config: ProductConfig,
    *,
    label: str,
    revenue_target: float,
    population: float,
    price: float,
    penetration: float,
) -> List[str]:
    issues: List[str] = []
    provided = [population > 0, price > 0, penetration > 0]
    if any(provided) and not all(provided):
        issues.append(
            f"{config.name}: patient-based {label} assumptions must include population, price, and penetration together."
        )
    if all(provided) and revenue_target > 0:
        implied_revenue = float(population) * float(price) * float(penetration)
        baseline = max(abs(implied_revenue), abs(float(revenue_target)), 1.0)
        if abs(implied_revenue - float(revenue_target)) / baseline > 0.10:
            issues.append(
                f"{config.name}: patient-based {label} revenue does not reconcile to the direct revenue target within 10%."
            )
    return issues


def validate_product_config(config: ProductConfig) -> List[str]:
    issues: List[str] = []
    if config.stage not in STAGE_SEQUENCE:
        issues.append(
            f"{config.name}: stage must be one of {', '.join(STAGE_SEQUENCE)}."
        )
    if not (0.0 <= config.success_prob <= 1.0):
        issues.append(f"{config.name}: success_prob must be between 0 and 1.")
    if config.stage == "Commercial" and config.time_to_market > 0:
        issues.append(f"{config.name}: Commercial stage requires time_to_market of 0.")
    if config.patent_years <= 0:
        issues.append(f"{config.name}: patent_years must be positive.")
    if config.time_to_market < 0 and not config.preexisting_market:
        issues.append(f"{config.name}: time_to_market must be >= 0 for non-preexisting products.")
    if config.sales_ramp_length is not None and config.sales_ramp_length < 0:
        issues.append(f"{config.name}: sales_ramp_length must be >= 0.")
    if config.sales_ramp_shape and config.sales_ramp_shape not in {"Linear", "S-curve", "Step"}:
        issues.append(f"{config.name}: sales_ramp_shape must be Linear, S-curve, or Step.")
    if config.launch_delay_sigma_years < 0:
        issues.append(f"{config.name}: launch_delay_sigma_years must be >= 0.")
    if config.upfront_timing not in {"from_start", "from_launch"}:
        issues.append(f"{config.name}: upfront_timing must be 'from_start' or 'from_launch'.")
    for label, value in {
        "cogs_patent": config.cogs_patent,
        "cogs_post": config.cogs_post,
        "labor_pct": config.labor_pct,
        "overhead_pct": config.overhead_pct,
        "material_pct": config.material_pct,
        "sales_marketing_pct": config.sales_marketing_pct,
        "gna_pct": config.gna_pct,
        "royalty_pct": config.royalty_pct,
        "rd_capitalization_ratio": config.rd_capitalization_ratio,
    }.items():
        if not (0.0 <= value <= 1.0):
            issues.append(f"{config.name}: {label} must be between 0 and 1.")
    for label, value in {
        "rd_remaining_pre_launch": config.rd_remaining_pre_launch,
        "rd_annual_post_launch": config.rd_annual_post_launch,
        "capex_remaining_pre_launch": config.capex_remaining_pre_launch,
        "capex_annual_post_launch": config.capex_annual_post_launch,
        "upfront_payment": config.upfront_payment,
        "patent_revenue_target": config.patent_revenue_target,
        "post_patent_revenue_target": config.post_patent_revenue_target,
    }.items():
        if value < 0:
            issues.append(f"{config.name}: {label} cannot be negative.")
    for label, value in {
        "rd_amort_years": config.rd_amort_years,
        "capex_dep_years": config.capex_dep_years,
    }.items():
        if value < 0:
            issues.append(f"{config.name}: {label} must be >= 0.")
    for label, value in {
        "patient_population_patent": config.patient_population_patent,
        "price_per_patient_patent": config.price_per_patient_patent,
        "penetration_patent": config.penetration_patent,
        "patient_population_post": config.patient_population_post,
        "price_per_patient_post": config.price_per_patient_post,
        "penetration_post": config.penetration_post,
    }.items():
        if value < 0:
            issues.append(f"{config.name}: {label} cannot be negative.")
    for label, value in {
        "market_growth_patent": config.market_growth_patent,
        "market_growth_post": config.market_growth_post,
    }.items():
        if value <= -1.0:
            issues.append(f"{config.name}: {label} must be greater than -1.")
    for key, prob in config.stage_transition_probabilities.items():
        if key not in _STAGE_TRANSITIONS:
            issues.append(f"{config.name}: stage transition '{key}' is not recognized.")
        if not (0.0 <= prob <= 1.0):
            issues.append(f"{config.name}: stage transition '{key}' must be between 0 and 1.")
    for stage, duration in config.stage_duration_years.items():
        if stage not in STAGE_SEQUENCE:
            issues.append(f"{config.name}: stage duration '{stage}' uses an unknown stage.")
        if duration < 0:
            issues.append(f"{config.name}: stage duration '{stage}' must be >= 0.")
    for stage, weight in config.stage_cost_weights.items():
        if stage not in STAGE_SEQUENCE:
            issues.append(f"{config.name}: stage cost weight '{stage}' uses an unknown stage.")
        if weight < 0:
            issues.append(f"{config.name}: stage cost weight '{stage}' must be >= 0.")
    for stage, weight in config.stage_capex_weights.items():
        if stage not in STAGE_SEQUENCE:
            issues.append(f"{config.name}: stage capex weight '{stage}' uses an unknown stage.")
        if weight < 0:
            issues.append(f"{config.name}: stage capex weight '{stage}' must be >= 0.")
    for stage, amount in config.trial_costs_by_phase.items():
        if stage not in STAGE_SEQUENCE:
            issues.append(f"{config.name}: trial cost '{stage}' uses an unknown stage.")
        if amount < 0:
            issues.append(f"{config.name}: trial cost '{stage}' must be >= 0.")
    for key, curve in config.stage_transition_curve.items():
        if key not in _STAGE_TRANSITIONS:
            issues.append(f"{config.name}: stage transition '{key}' curve is not recognized.")
        for prob in curve:
            if not (0.0 <= prob <= 1.0):
                issues.append(
                    f"{config.name}: stage transition '{key}' curve values must be between 0 and 1."
                )
    for factor in config.post_patent_erosion:
        if not (0.0 <= factor <= 1.0):
            issues.append(f"{config.name}: post_patent_erosion values must be between 0 and 1.")
    normalized_milestones: List[Milestone] = []
    for milestone in config.milestones:
        if isinstance(milestone, Milestone):
            normalized_milestones.append(milestone)
            continue
        try:
            normalized_milestones.append(Milestone(**milestone))
        except (TypeError, ValueError, KeyError):
            issues.append(f"{config.name}: milestone entries must include name, year_offset, and amount.")
    for milestone in normalized_milestones:
        if milestone.amount < 0:
            issues.append(f"{config.name}: milestone '{milestone.name}' amount cannot be negative.")
        if not (0.0 <= milestone.probability <= 1.0):
            issues.append(f"{config.name}: milestone '{milestone.name}' probability must be 0-1.")
        if milestone.timing not in {"from_start", "from_launch"}:
            issues.append(
                f"{config.name}: milestone '{milestone.name}' timing must be 'from_start' or 'from_launch'."
            )
    issues.extend(
        _validate_patient_target_block(
            config,
            label="patent-period",
            revenue_target=config.patent_revenue_target,
            population=config.patient_population_patent,
            price=config.price_per_patient_patent,
            penetration=config.penetration_patent,
        )
    )
    issues.extend(
        _validate_patient_target_block(
            config,
            label="post-patent",
            revenue_target=config.post_patent_revenue_target,
            population=config.patient_population_post,
            price=config.price_per_patient_post,
            penetration=config.penetration_post,
        )
    )
    if config.stage in STAGE_SEQUENCE and config.stage_duration_years and not config.preexisting_market:
        stage_idx = STAGE_SEQUENCE.index(config.stage)
        forward_stages = STAGE_SEQUENCE[stage_idx:-1]
        duration_sum = sum(int(config.stage_duration_years.get(stage, 0)) for stage in forward_stages)
        if duration_sum > 0 and duration_sum != int(max(config.time_to_market, 0)):
            issues.append(
                f"{config.name}: stage durations from {config.stage} to launch must sum to time_to_market."
            )

        active_weight_stages = [stage for stage in forward_stages if int(config.stage_duration_years.get(stage, 0)) > 0]
        if active_weight_stages and config.stage_cost_weights:
            cost_weight_total = sum(float(config.stage_cost_weights.get(stage, 0.0)) for stage in active_weight_stages)
            if cost_weight_total <= 0:
                issues.append(
                    f"{config.name}: stage cost weights across active pre-launch stages must allocate some positive weight."
                )
        if active_weight_stages and config.stage_capex_weights:
            capex_weight_total = sum(float(config.stage_capex_weights.get(stage, 0.0)) for stage in active_weight_stages)
            if capex_weight_total <= 0:
                issues.append(
                    f"{config.name}: stage capex weights across active pre-launch stages must allocate some positive weight."
                )
    return issues


def validate_portfolio(portfolio: Portfolio) -> List[str]:
    issues: List[str] = validate_model_config(portfolio.model_config)
    for product in portfolio.products:
        issues.extend(validate_product_config(product.config))
    return issues


if __name__ == "__main__":
    model_cfg = ModelConfig()
    moonshine_cfg = ProductConfig(
        name="Vaccine_Moonshine",
        stage="Market",
        success_prob=1.0,
        include_in_consolidation=True,
        preexisting_market=True,
        time_to_market=-20,
        patent_years=20,
        patent_revenue_target=11_250_000.0,
        post_patent_revenue_target=6_300_000.0,
        market_growth_patent=0.005,
        market_growth_post=0.0,
        cogs_patent=0.30,
        cogs_post=0.50,
        sales_marketing_pct=0.15,
        gna_pct=0.10,
        rd_annual_post_launch=500_000.0,
        capex_annual_post_launch=100_000.0,
    )

    moonshine = Product(moonshine_cfg, model_cfg)
    portfolio = Portfolio([moonshine], model_cfg)

    engine = ValuationEngine(portfolio)
    val_res = engine.run()
    print("rNPV:", round(val_res.rnpv, 2), model_cfg.currency)

    vc_inputs = VCInputs(
        exit_year=model_cfg.first_year + 8,
        target_irr=0.40,
        investor_ownership_at_exit=0.25,
        new_money=20_000_000.0,
    )
    vc_val = VCValuator(val_res)
    vc_result = vc_val.vc_method(vc_inputs, exit_multiple=10.0)
    print("VC implied pre-money:", round(vc_result["implied_pre_money"], 2))

    drought = Scenario(
        name="Drought",
        revenue_multiplier=0.8,
        cost_multiplier=1.1,
        discount_rate_shift=0.02,
        success_prob_multiplier=0.9,
    )
    disease = Scenario(
        name="Disease",
        revenue_multiplier=0.7,
        cost_multiplier=1.15,
        discount_rate_shift=0.03,
        success_prob_multiplier=0.8,
    )

    scen_engine = ScenarioEngine(portfolio)
    scen_df = scen_engine.run_scenarios([drought, disease], ebitda_year_offset=5)
    print("\nScenario comparison:")
    print(scen_df)

    mc = MonteCarloEngine(portfolio)
    sims = mc.simulate(n_sims=200, revenue_sigma=0.15, cost_sigma=0.10, random_seed=42)
    print("\nMonte Carlo mean rNPV:", sims.mean())
    print("95% VaR:", MonteCarloEngine.value_at_risk(sims, alpha=0.95))
    print("95% CVaR:", MonteCarloEngine.conditional_value_at_risk(sims, alpha=0.95))

    idx = pd.period_range("2015", "2023", freq="Y").to_timestamp()
    hist_prices = pd.Series(
        [100, 102, 105, 108, 110, 115, 118, 120, 123],
        index=idx,
        name="price",
    )
    fe = ForecastEngine(model_cfg)
    price_fc = fe.forecast_arima(hist_prices, order=(1, 1, 1), steps=model_cfg.n_years)
    products_forecasted = fe.apply_price_forecast_to_products(
        portfolio.products,
        price_fc,
        base_price=hist_prices.iloc[-1],
        mode="growth",
        growth_scale=1.0,
    )
    port_fc = Portfolio(products_forecasted, model_cfg)
    val_engine_fc = ValuationEngine(port_fc)
    val_res_fc = val_engine_fc.run()
    print("rNPV with price-linked growth:", round(val_res_fc.rnpv, 2), model_cfg.currency)

    dates = pd.date_range("2015-01-01", periods=9, freq="Y")
    hist_prices_df = pd.DataFrame({"ds": dates, "y": [100, 102, 105, 108, 110, 115, 118, 120, 123]})
    bridge = ForecastScenarioBridge(portfolio, fe)
    scen_results = bridge.build_price_scenarios_from_prophet(hist_prices_df, freq="Y", ebitda_year_offset=5)
    print("\nProphet-driven scenario comparison:")
    print(scen_results)
