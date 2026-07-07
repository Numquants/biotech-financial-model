import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
