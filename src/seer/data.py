"""Domain/dataset interface for the L0/L1/L2 transfer ladder (ROADMAP.md, TRAINING.md section 5).

The core empirical question SEER exists to answer is a *transfer* question: does
the energy signal beat the token probe out-of-domain? That means every dataset
here is tagged with a domain, and evaluation always compares a train domain
against one or more held-out transfer domains — never a single in-domain number.

Two requirements baked into the interface, both learned the hard way per
TRAINING.md:

- Section 3: dense, per-step, state-dependent supervision. A single terminal
  target does not train state-tracking. ``StateTrackingExample.step_targets``
  is a target per step, not one target for the whole sequence.
- Section 5: a single self-certainty head is domain-specific, not universal.
  ``Domain`` is a first-class tag threaded through datasets and eval so results
  are never silently pooled across domains.

No real data is bundled. ``SyntheticStateTrackingDataset`` exists only so the
training loop and eval harness have something concrete to run against locally,
without a download, for smoke testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class Domain:
    """A named data/task domain (e.g. "arithmetic", "entity_tracking", "logic_puzzles").

    Kept as a lightweight tag rather than an enum so new domains don't require
    editing this module — but the name is what eval groups by, so keep it
    consistent across a given experiment.
    """

    name: str
    description: str = ""


@dataclass(slots=True)
class StateTrackingExample:
    """One dense-supervision example: an input sequence plus a target at every step.

    Attributes:
        input_ids: Token ids, shape ``(seq_len,)``.
        step_targets: Target label at every step, shape ``(seq_len,)``. Per
            TRAINING.md section 3, loss must be computed at (a superset of)
            these positions, not just the final one.
        step_mask: 1 where ``step_targets`` is a real supervised position, 0
            where it should be ignored (e.g. padding, or steps with no defined
            ground truth yet). Shape ``(seq_len,)``.
        domain: Which domain this example belongs to.
    """

    input_ids: Tensor
    step_targets: Tensor
    step_mask: Tensor
    domain: Domain


class DomainDataset(Dataset, ABC):
    """Abstract dataset that knows its own domain.

    Subclasses implement real task/domain data loading. This ABC exists so the
    training loop and eval harness can depend on ``.domain`` without caring how
    a given dataset produces examples.
    """

    domain: Domain

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> StateTrackingExample: ...


@dataclass(slots=True)
class TransferLadder:
    """Bundles a train domain with one or more held-out transfer (OOD) domains.

    This is the unit experiment 1 actually runs (ROADMAP.md experiment ladder,
    step 1): train the energy head on ``train`` and evaluate AUROC/ECE on each
    entry of ``transfer`` — the number that decides H-energy-transfer.
    """

    train: DomainDataset
    transfer: list[DomainDataset]

    def all_domains(self) -> list[Domain]:
        return [self.train.domain] + [d.domain for d in self.transfer]


@dataclass(slots=True)
class StateTrackingBatch:
    """Stacked batch of :class:`StateTrackingExample` for the training loop."""

    input_ids: Tensor
    step_targets: Tensor
    step_mask: Tensor
    domain: Domain


def collate_state_tracking_examples(examples: list[StateTrackingExample]) -> StateTrackingBatch:
    """Stack a list of same-domain, same-length examples into one batch.

    Raises if the batch mixes domains — TRAINING.md section 5's domain-matching
    rule means a batch silently mixing domains would corrupt what "train domain"
    even means for the run.
    """
    domains = {ex.domain.name for ex in examples}
    if len(domains) > 1:
        raise ValueError(f"batch mixes domains {domains}; a batch must be single-domain")
    return StateTrackingBatch(
        input_ids=torch.stack([ex.input_ids for ex in examples]),
        step_targets=torch.stack([ex.step_targets for ex in examples]),
        step_mask=torch.stack([ex.step_mask for ex in examples]),
        domain=examples[0].domain,
    )


class SyntheticStateTrackingDataset(DomainDataset):
    """Synthetic local smoke-test dataset — NOT a real domain, just wiring.

    Task: track a running parity (XOR) of a random binary sequence and predict
    the running parity at every step (dense supervision by construction). This
    exists purely so ``train.py`` / ``eval.py`` have something deterministic and
    fast to run against on CPU without a download; it is not a substitute for a
    real domain in the transfer ladder.
    """

    def __init__(
        self,
        domain: Domain,
        num_examples: int,
        seq_len: int = 32,
        vocab_size: int = 2,
        seed: int = 0,
    ) -> None:
        self.domain = domain
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        generator = torch.Generator().manual_seed(seed)
        self._bits = torch.randint(
            0, vocab_size, (num_examples, seq_len), generator=generator
        )

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> StateTrackingExample:
        bits = self._bits[index]
        running_parity = torch.remainder(torch.cumsum(bits, dim=0), 2)
        mask = torch.ones(self.seq_len, dtype=torch.bool)
        return StateTrackingExample(
            input_ids=bits,
            step_targets=running_parity,
            step_mask=mask,
            domain=self.domain,
        )
