import contextlib
import unittest
from unittest.mock import patch

import pandas as pd

import streamlit_app


class EditWorkflowHelperTests(unittest.TestCase):
    def test_required_row_fields_falls_back_to_identifier_like_column(self) -> None:
        columns = ["Pool name", "Allocation method", "Amount"]
        required_fields = streamlit_app._required_row_fields(
            columns,
            id_column=None,
            name_column=None,
            initial_values={"Pool name": "", "Allocation method": "Equal", "Amount": 0.0},
        )
        self.assertEqual(required_fields, ["Pool name"])

    def test_validate_row_form_values_blocks_blank_required_fields(self) -> None:
        errors = streamlit_app._validate_row_form_values(
            {"Pool name": "   ", "Amount": 5.0},
            ["Pool name"],
        )
        self.assertEqual(errors, ["Pool name is required."])

    def test_prime_row_form_state_refreshes_widgets_for_new_selection(self) -> None:
        session_state = {}
        with patch.object(streamlit_app.st, "session_state", session_state):
            streamlit_app._prime_row_form_state(
                "uses_table",
                "edit",
                ["Item", "Amount"],
                {"Item": "Utilities", "Amount": 12.0},
                source_token=0,
            )
            streamlit_app._prime_row_form_state(
                "uses_table",
                "edit",
                ["Item", "Amount"],
                {"Item": "Raw materials", "Amount": 24.0},
                source_token=1,
            )

        self.assertEqual(session_state["uses_table_edit_Item"], "Raw materials")
        self.assertEqual(session_state["uses_table_edit_Amount"], 24.0)
        self.assertEqual(session_state["uses_table_edit_source"], 1)

    def test_clear_row_form_state_removes_seeded_widget_state(self) -> None:
        session_state = {
            "uses_table_edit_Item": "Utilities",
            "uses_table_edit_Amount": 12.0,
            "uses_table_edit_source": 0,
        }
        with patch.object(streamlit_app.st, "session_state", session_state):
            streamlit_app._clear_row_form_state("uses_table", "edit", ["Item", "Amount"])

        self.assertNotIn("uses_table_edit_Item", session_state)
        self.assertNotIn("uses_table_edit_Amount", session_state)
        self.assertNotIn("uses_table_edit_source", session_state)

    def test_clear_legacy_row_form_state_preserves_active_edit_workflow_flag(self) -> None:
        active_edit_key = streamlit_app._panel_state_key("uses_table", "edit")
        session_state = {
            "uses_table_add_open": True,
            "uses_table_edit_open": True,
            active_edit_key: True,
        }

        streamlit_app._clear_legacy_row_form_widget_state(session_state)

        self.assertNotIn("uses_table_add_open", session_state)
        self.assertNotIn("uses_table_edit_open", session_state)
        self.assertTrue(session_state[active_edit_key])

    def test_debt_schedule_edit_persists_when_editor_returns_stale_frame(self) -> None:
        old_df = pd.DataFrame(
            [
                {
                    "Year": 2024,
                    "Debt drawdowns": 60_000_000.0,
                    "Loan tenor (years)": 25.0,
                    "Manual debt repayments": 0.0,
                },
                {
                    "Year": 2025,
                    "Debt drawdowns": 20_000_000.0,
                    "Loan tenor (years)": 24.0,
                    "Manual debt repayments": 0.0,
                },
                {
                    "Year": 2026,
                    "Debt drawdowns": 20_000_000.0,
                    "Loan tenor (years)": 23.0,
                    "Manual debt repayments": 0.0,
                },
            ]
        )
        updated_row = {
            "Year": 2026.0,
            "Debt drawdowns": 45_000_000.0,
            "Loan tenor (years)": 23.0,
            "Manual debt repayments": 0.0,
        }
        session_state = {
            "debt_schedule_table": old_df.copy(),
            streamlit_app._panel_state_key("debt_schedule_table", "edit"): True,
            "debt_schedule_table_edit_row_select": 2026,
            "debt_schedule_table_edit_Year": 2026.0,
            "debt_schedule_table_edit_Debt drawdowns": 20_000_000.0,
            "debt_schedule_table_edit_Loan tenor (years)": 23.0,
            "debt_schedule_table_edit_Manual debt repayments": 0.0,
            "debt_schedule_table_edit_source": 2026,
        }
        action_cols = [contextlib.nullcontext() for _ in range(4)]
        editor_keys = []

        def fake_data_editor(df, **kwargs):
            editor_keys.append(kwargs["key"])
            return old_df.copy()

        with patch.object(streamlit_app.st, "session_state", session_state):
            with patch.object(streamlit_app.st, "success"), patch.object(streamlit_app.st, "caption"), patch.object(
                streamlit_app.st, "markdown"
            ):
                with patch.object(streamlit_app.st, "columns", return_value=action_cols):
                    with patch.object(streamlit_app, "_render_row_selector", return_value=2):
                        with patch.object(streamlit_app, "_render_edit_button"), patch.object(
                            streamlit_app, "_add_row_via_form", side_effect=lambda *args, **kwargs: args[1]
                        ), patch.object(
                            streamlit_app, "_remove_selected_row", side_effect=lambda *args, **kwargs: args[1]
                        ), patch.object(
                            streamlit_app, "_apply_yearly_increment", side_effect=lambda *args, **kwargs: args[1]
                        ):
                            with patch.object(streamlit_app.st, "selectbox", return_value=2026), patch.object(
                                streamlit_app.st, "rerun"
                            ), patch.object(
                                streamlit_app, "_render_row_form", return_value=("save", updated_row)
                            ), patch.object(
                                streamlit_app.st, "data_editor", side_effect=fake_data_editor
                            ), patch.object(
                                streamlit_app, "_validate_selection"
                            ):
                                result_df = streamlit_app._render_product_assumption_table(
                                    session_key="debt_schedule_table",
                                    default_factory=lambda: old_df.copy(),
                                    blank_row_factory=lambda df: {
                                        "Year": 2027,
                                        "Debt drawdowns": 0.0,
                                        "Loan tenor (years)": 1.0,
                                        "Manual debt repayments": 0.0,
                                    },
                                    id_column=None,
                                    name_column="Year",
                                )

        self.assertEqual(editor_keys, ["debt_schedule_table_editor_1"])
        self.assertEqual(float(result_df.loc[2, "Debt drawdowns"]), 45_000_000.0)
        self.assertEqual(float(session_state["debt_schedule_table"].loc[2, "Debt drawdowns"]), 45_000_000.0)
        self.assertEqual(session_state["debt_schedule_table_row_notice"], "Row updated")
        self.assertNotIn("debt_schedule_table_editor_skip_writeback", session_state)


if __name__ == "__main__":
    unittest.main()
