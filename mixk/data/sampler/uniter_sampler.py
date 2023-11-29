"""sampler for length bucketing (batch by tokens)"""
import random
from torch.utils.data import Sampler
from cytoolz import partition_all
from .builder import BATCH_SAMPLER
from torch import distributed as dist


@BATCH_SAMPLER.register_module()
class TokenBucketSampler(Sampler):

    def __init__(
        self,
        lens,
        bucket_size,
        batch_size,
        drop_last=False,
        size_multiple=8,
        num_replicas=None,
        rank=None,
    ):
        if num_replicas is None:
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()

        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = drop_last
        self.is_dist = dist.is_initialized()
        self._size_mul = size_multiple
        self.num_replicas = num_replicas
        self.rank = rank
        self.getbatches()
        self._reset = True

        # world_size must be evenly divisible by self._size_mul
        if self.is_dist:
            assert self._size_mul % self.num_replicas == 0

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        if self._reset:
            self._reset = False
        else:
            self.getbatches()
            # print("*********** reset batch sampler iter ***********")
        return iter(self.batches)

    def getbatches(self):
        self.batches = []
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [
            sorted(ids[i:i + self._bucket_size], key=self._sort_fn, reverse=True)
            for i in range(0, len(ids), self._bucket_size)
        ]
        # fill batches until max_token (include padding)
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul) > self._max_tok):
                    if not batch_indices:
                        raise ValueError('max_tokens too small / max_seq_len too long')
                    assert len(batch_indices) % self._size_mul == 0
                    if self.is_dist:
                        # subsample
                        subindices = batch_indices[self.rank:len(batch_indices):self.num_replicas]
                        self.batches.append(subindices)
                    else:
                        self.batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                if self.is_dist:
                    # subsample
                    subindices = batch_indices[self.rank:len(batch_indices):self.num_replicas]
                    self.batches.append(subindices)
                else:
                    self.batches.append(batch_indices)
        random.shuffle(self.batches)
        # for test
        # self.batches = self.batches[:16]
        return self.batches

    def __len__(self):
        if not self._reset:
            self.getbatches()
            self._reset = True

        return len(self.batches)
