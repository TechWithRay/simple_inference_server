from __future__ import annotations

import types
from typing import Any


class FakeTensor:
    """A minimal tensor stub that mimics torch.Tensor for testing."""

    def __init__(self, shape: tuple[int, ...], device: str = "cpu") -> None:
        self.shape = shape
        self.device = types.SimpleNamespace(type=device)
        self.dtype = "long"

    def dim(self) -> int:
        return len(self.shape)

    def unsqueeze(self, dim: int) -> FakeTensor:
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return FakeTensor(tuple(new_shape), self.device.type)

    def view(self, *shape: int) -> FakeTensor:
        return FakeTensor(shape, self.device.type)

    def to(self, device: Any, **kwargs: Any) -> FakeTensor:
        return FakeTensor(self.shape, device)

    def pin_memory(self) -> FakeTensor:
        return self

    def tolist(self) -> list[int]:
        return [0] * self.shape[0]

    def __getitem__(self, idx: Any) -> FakeTensor:
        tuple_len_slice = 2
        if isinstance(idx, tuple) and len(idx) == tuple_len_slice and isinstance(idx[1], slice):
            start = idx[1].start if idx[1].start is not None else 0
            stop = idx[1].stop if idx[1].stop is not None else self.shape[1]
            new_width = max(0, stop - start)
            return FakeTensor((self.shape[0], new_width), self.device.type)
        return self


class InferenceModeContext:
    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
