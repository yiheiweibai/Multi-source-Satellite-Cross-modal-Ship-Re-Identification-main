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


class CrossModalPKSampler(Sampler[int]):
    """模态均衡 PK 采样器（兼容 PKSampler 参数 p/k/seed）。

    每个 batch 选 P 个身份、每身份 K 张，且 K 张内强制模态均衡：
    - K 张优先 O/S 各半（K 为奇数时 O 侧多 1 张）；
    - 某模态样本不足时用另一模态剩余样本凑足，全部不足才允许重复样本（维持 batch 尺寸）；
    - 优先选择同时含双模态样本的身份，保证批内跨模态 positive pair 充足。
    跨模态损失（CrossModalInfoNCE / CrossModalHardTriplet）依赖批内 O/S 对齐，
    本采样器是其数据侧配套。
    """

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

        # 按身份记录 O/S 样本索引；同时含双模态样本的身份优先
        self.mod_by_label: Dict[int, tuple[List[int], List[int]]] = {}
        self.dual_labels: List[int] = []
        if modalities is not None:
            mod = list(modalities)
            for lab in self.valid_labels:
                idxs = self.indices_by_label[lab]
                o_idxs = [i for i in idxs if mod[i] == 0]
                s_idxs = [i for i in idxs if mod[i] == 1]
                self.mod_by_label[lab] = (o_idxs, s_idxs)
                if o_idxs and s_idxs:
                    self.dual_labels.append(lab)
        else:
            # modalities 缺失时退化为普通 PK（每身份全量索引）
            for lab in self.valid_labels:
                self.mod_by_label[lab] = (list(self.indices_by_label[lab]), [])

    def __len__(self) -> int:
        return (self.n + self.p * self.k - 1) // (self.p * self.k) * self.p * self.k

    def __iter__(self) -> Iterator[int]:
        self.rng = random.Random()
        num_batches = max(1, (self.n + self.p * self.k - 1) // (self.p * self.k))
        for _ in range(num_batches):
            pool = self.dual_labels if self.dual_labels else self.valid_labels
            labs = self.rng.sample(pool, min(self.p, len(pool)))
            for lab in labs:
                for idx in self._sample_identity(lab):
                    yield idx

    def _sample_identity(self, lab: int) -> List[int]:
        """单身份采样：K 张内尽量 O/S 各半，不足时用另一模态/重复样本凑足。"""
        o_idxs, s_idxs = self.mod_by_label.get(
            lab, (list(self.indices_by_label[lab]), [])
        )
        picked: List[int] = []
        half = self.k // 2
        o_pick = self.rng.sample(o_idxs, min(half, len(o_idxs)))
        s_pick = self.rng.sample(s_idxs, min(half, len(s_idxs)))
        # K 为奇数时补 1 张：优先补 O 侧，其次 S 侧
        if self.k % 2 == 1:
            o_rem = [i for i in o_idxs if i not in o_pick]
            if o_rem:
                o_pick += self.rng.sample(o_rem, 1)
            else:
                s_rem = [i for i in s_idxs if i not in s_pick]
                if s_rem:
                    s_pick += self.rng.sample(s_rem, 1)
        picked = o_pick + s_pick
        # 不足 K：优先从另一模态剩余中补，仍不足则允许重复样本（维持 batch 尺寸稳定）
        all_idxs = o_idxs + s_idxs
        remaining = [i for i in all_idxs if i not in picked]
        while len(picked) < self.k and remaining:
            picked.append(self.rng.choice(remaining))
            remaining = [i for i in all_idxs if i not in picked]
        while len(picked) < self.k:
            picked.append(self.rng.choice(all_idxs))
        return picked
