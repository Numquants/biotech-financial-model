"""Helpers for deferred Streamlit widget state in editable biotech tables."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def _pending_selection_key(select_key: str) -> str:
    return f"{select_key}_pending"


def _panel_state_key(section_key: str, panel_name: str) -> str:
    return f"{section_key}_{panel_name}_open"


def _pending_panel_state_key(panel_key: str) -> str:
    return f"{panel_key}_pending"


def _set_pending_state(pending_key: str, value: object) -> None:
    st.session_state[pending_key] = value


def _consume_pending_state(pending_key: str, state_key: str) -> Optional[object]:
    if pending_key not in st.session_state:
        return None
    value = st.session_state.pop(pending_key)
    st.session_state[state_key] = value
    return value


def _set_pending_selection(select_key: str, value: Optional[object]) -> None:
    _set_pending_state(_pending_selection_key(select_key), value)


def _consume_pending_selection(select_key: str) -> Optional[object]:
    return _consume_pending_state(_pending_selection_key(select_key), select_key)


def _set_pending_panel_state(panel_key: str, value: bool) -> None:
    _set_pending_state(_pending_panel_state_key(panel_key), value)


def _consume_pending_panel_state(panel_key: str) -> Optional[bool]:
    value = _consume_pending_state(_pending_panel_state_key(panel_key), panel_key)
    if value is None:
        return None
    coerced = bool(value)
    st.session_state[panel_key] = coerced
    return coerced


def _row_identifier(df: pd.DataFrame, idx: int, id_column: Optional[str]) -> object:
    if id_column and id_column in df.columns:
        value = df.at[idx, id_column]
        if pd.isna(value):
            return idx
        return value
    return idx


def _resolve_selected_index_from_value(
    df: pd.DataFrame,
    selected_id: Optional[object],
    id_column: Optional[str],
) -> Optional[int]:
    if df.empty:
        return None
    if id_column and id_column in df.columns and selected_id is not None:
        matches = df.index[df[id_column] == selected_id]
        if len(matches):
            return matches[0]
    if selected_id in df.index:
        return selected_id
    return df.index[0]


def _resolve_selected_index(
    df: pd.DataFrame,
    select_key: str,
    id_column: Optional[str],
) -> Optional[int]:
    return _resolve_selected_index_from_value(df, st.session_state.get(select_key), id_column)


def _validate_selection(
    df: pd.DataFrame,
    select_key: str,
    id_column: Optional[str],
) -> None:
    if df.empty:
        _set_pending_selection(select_key, None)
        return

    selected_idx = _resolve_selected_index(df, select_key, id_column)
    if selected_idx is None:
        _set_pending_selection(select_key, _row_identifier(df, df.index[0], id_column))
        return

    selected_id = _row_identifier(df, selected_idx, id_column)
    if selected_id != st.session_state.get(select_key):
        _set_pending_selection(select_key, selected_id)

