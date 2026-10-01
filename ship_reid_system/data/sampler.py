"""PK 采样器：每个 batch 含 P 个身份、每个身份 K 张图像。

跨模态 ReID 训练要求批内同时出现光学与 SAR 样本，本采样器默认优先选取
"同时拥有两种模态样本"的身份，保证批内跨模态正/负样本充足。
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence

from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    def __init__(
        self,
        labels: Sequence[int],
        p: int = 8,
        k: int = 4,
        modalities: Optional[Sequence[int]] = None,
        prefer_dual_modal: bool = True,
        seed: int = 0,
    ) -> None:
        self.p = p
        self.k = k
        self.labels = list(labels)
        self.n = len(self.labels)
        self.rng = random.Random(seed)

        self.indices_by_label: Dict[int, List[int]] = defaultdict(list)
        for i, lab in enumerate(self.labels):
            self.indices_by_label[lab].append(i)

        self.valid_labels = [lab for lab, idxs in self.indices_by_label.items() if len(idxs) > 0]

        # 同时含两种模态样本的身份（跨模态训练优先）
        self.dual_labels: List[int] = []
        if prefer_dual_modal and modalities is not None:
            mod = list(modalities)
            for lab in self.valid_labels:
                mset = {mod[i] for i in self.indices_by_label[lab]}
                if len(mset) >= 2:
                    self.dual_labels.append(lab)

    def __len__(self) -> int:
        return (self.n + self.p * self.k - 1) // (self.p * self.k) * self.p * self.k

    def __iter__(self) -> Iterator[int]:
        self.rng = random.Random()
        num_batches = max(1, (self.n + self.p * self.k - 1) // (self.p * self.k))
        for _ in range(num_batches):
            pool = self.dual_labels if self.dual_labels else self.valid_labels
            labs = self.rng.sample(pool, min(self.p, len(pool)))
            for lab in labs:
                idxs = self.indices_by_label[lab]
                picked = self.rng.sample(idxs, min(self.k, len(idxs)))
                # 不足 K 时随机补足（同一身份重复样本，维持 batch 尺寸稳定）
                while len(picked) < self.k:
                    picked.append(self.rng.choice(idxs))
                yield from picked
