"""Type definitions for QoL."""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

BytePair: TypeAlias = tuple[bytes, bytes]
ByteSequence: TypeAlias = Sequence[bytes]
MergeList: TypeAlias = Sequence[BytePair]
Vocab: TypeAlias = Mapping[int, bytes]
