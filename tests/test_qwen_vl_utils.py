from __future__ import annotations

import threading
from typing import Any

import pytest

# Check if we are running with the conftest-patched torch
import torch

from app.models.base import ChatGeneration
from app.models.qwen_vl import QwenVLChat

# --- Reusable Mocks ---------------------------------------------------------

class MockTokenizer:
    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2, 3] if s else []

    def batch_decode(self, ids: Any, skip_special_tokens: bool = True) -> list[str]:
        return ["generated text"]


class MockProcessor:
    def __init__(self) -> None:
        self.tokenizer = MockTokenizer()

    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
    ) -> dict[str, Any]:
        assert tokenize is True
        assert return_tensors == "pt"
        assert return_dict is True
        return {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            # Include a non-tensor field to verify normalization logic
            "metadata": "should be dropped",
        }

    def batch_decode(self, ids: Any, skip_special_tokens: bool = True) -> list[str]:
        return ["generated text"]


class MockModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu") if hasattr(torch, "device") else "cpu"

    def generate(self, **kwargs: Any) -> Any:
        # Return a FakeTensor with shape (batch=1, total_len=5)
        return torch.full((1, 5), 0)


# --- Fixtures ---------------------------------------------------------------

# mock_torch fixture is provided by conftest.py

# --- Tests ------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "stop", "expected", "hit"),
    [
        ("hello stop there", ["stop"], "hello", True),
        ("no match here", ["stop"], "no match here", False),
        ("abc END xyz", ["END", "STOP"], "abc", True),
        ("abc END xyz stop", ["stop"], "abc END xyz", True),
    ],
)
def test_trim_with_stop(text: str, stop: list[str], expected: str, hit: bool) -> None:
    trimmed, got_hit = QwenVLChat._trim_with_stop(text, stop)
    assert trimmed == expected
    assert got_hit is hit


def test_prepare_inputs_drops_non_tensor_fields(mock_torch: None) -> None:
    obj = QwenVLChat.__new__(QwenVLChat)
    obj.processor = MockProcessor()

    prepared, prompt_len = obj.prepare_inputs(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )

    expected_len = 3
    assert prompt_len == expected_len
    # Non-tensor metadata should have been dropped by _normalize_chat_template_output
    assert "metadata" not in prepared
    assert "input_ids" in prepared
    assert "attention_mask" in prepared
    # Verify values are FakeTensors
    assert torch.is_tensor(prepared["input_ids"])


def test_generate_prepared_handles_normalized_inputs(mock_torch: None) -> None:
    obj = QwenVLChat.__new__(QwenVLChat)
    obj.processor = MockProcessor()
    obj.model = MockModel()
    obj._gen_lock = threading.Lock()

    # Create clean inputs (simulating output of prepare_inputs)
    prepared = {
        "input_ids": torch.arange(3).unsqueeze(0),
        "attention_mask": torch.ones((1, 3)),
        "_prompt_len": 3,
    }

    max_new_tokens = 2
    result = obj.generate_prepared(
        prepared,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop=None,
        cancel_event=None,
    )

    assert isinstance(result, ChatGeneration)
    expected_prompt_tokens = 3
    assert result.prompt_tokens == expected_prompt_tokens
    assert result.completion_tokens == max_new_tokens
    assert result.text == "generated text"
