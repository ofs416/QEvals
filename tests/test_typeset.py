import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typeset import _merge, typeset_one


def test_merge_pairs_html_to_optimiser_metadata_by_index():
    opt = [{"commandWord": "Find", "marks": 3, "difficulty": "higher"}]
    html = [{"question_html": "<p>q</p>", "mark_scheme_html": "<ol></ol>"}]
    merged = _merge(opt, html)
    assert merged[0]["commandWord"] == "Find"
    assert merged[0]["marks"] == 3
    assert merged[0]["question_html"] == "<p>q</p>"


def test_merge_tolerates_missing_html_entry():
    opt = [{"commandWord": "Find", "marks": 2, "difficulty": "foundation"}]
    merged = _merge(opt, [])  # typesetter dropped the question
    assert merged[0]["question_html"] == ""
    assert merged[0]["marks"] == 2  # metadata still carried


@pytest.mark.asyncio
async def test_typeset_one_skips_failed_optimisation():
    opt = {"generation_id": "g1", "skipped": True, "parse_ok": False, "questions": []}
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock) as call:
        rec = await typeset_one(opt)
        call.assert_not_called()
    assert rec["skipped"] is True
    assert rec["questions"] == []
