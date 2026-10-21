from __future__ import annotations

import unittest

import torch

from specforge.algorithms.common.providers import OfflineCaptureLayout
from specforge.training.local_online import (
    ColocatedOnlineDataLoader,
    distributed_prompt_indices,
)


class _Capture:
    def __init__(self, *, hidden_size: int = 3) -> None:
        self.hidden_size = hidden_size
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.calls: list[list[list[int]]] = []
        self.closed = False

    def capture_rows(self, rows):
        rows = [list(row) for row in rows]
        self.calls.append(rows)
        aux = [
            torch.full(
                (len(row), self.hidden_size),
                float(row[0]),
                device=self.device,
            )
            for row in rows
        ]
        last = [tensor + 1000 for tensor in aux]
        return aux, last

    def close(self) -> None:
        self.closed = True


def _prompts(size: int):
    return [
        {
            "payload": {
                "input_ids": [index, index + 100],
                "loss_mask": [0, 1],
            }
        }
        for index in range(size)
    ]


def _collate(samples):
    return {
        key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]
    }


def _loader(
    *, prompts, tp_rank, dp_rank, include_last=False, sampler_drop_last=True
):
    capture = _Capture()
    layout = OfflineCaptureLayout(
        capture_method="dspark" if include_last else "dflash",
        aux_feature="hidden_states",
        last_hidden_feature=("target_last_hidden_states" if include_last else None),
        passthrough=(
            ("input_ids", "input_ids"),
            ("loss_mask", "loss_mask"),
        ),
    )
    loader = ColocatedOnlineDataLoader(
        prompts=prompts,
        capture=capture,
        capture_layout=layout,
        collate_fn=_collate,
        strategy="dspark" if include_last else "dflash",
        batch_size=2,
        tp_size=2,
        tp_rank=tp_rank,
        dp_size=2,
        dp_rank=dp_rank,
        seed=19,
        sampler_drop_last=sampler_drop_last,
    )
    return loader, capture


class ColocatedOnlineDataLoaderTest(unittest.TestCase):
    def test_tp_peers_capture_identical_batches_then_keep_disjoint_rows(self):
        prompts = _prompts(24)
        tp0, capture0 = _loader(prompts=prompts, tp_rank=0, dp_rank=0)
        tp1, capture1 = _loader(prompts=prompts, tp_rank=1, dp_rank=0)

        batches0 = list(tp0)
        batches1 = list(tp1)

        self.assertEqual(capture0.calls, capture1.calls)
        self.assertEqual(len(batches0), 3)
        self.assertEqual(tp0.dataset_size, 6)
        for left, right, captured in zip(batches0, batches1, capture0.calls):
            expected = [f"local:{row[0]}" for row in captured]
            self.assertEqual(left.sample_ids, expected[:2])
            self.assertEqual(right.sample_ids, expected[2:])
            self.assertTrue(set(left.sample_ids).isdisjoint(right.sample_ids))
            self.assertEqual(tuple(left.tensors["hidden_states"].shape), (2, 2, 3))

    def test_dp_groups_receive_different_deterministic_prompt_shards(self):
        prompts = _prompts(24)
        dp0, capture0 = _loader(prompts=prompts, tp_rank=0, dp_rank=0)
        dp1, capture1 = _loader(prompts=prompts, tp_rank=0, dp_rank=1)

        list(dp0)
        list(dp1)

        rows0 = {row[0] for call in capture0.calls for row in call}
        rows1 = {row[0] for call in capture1.calls for row in call}
        self.assertTrue(rows0.isdisjoint(rows1))
        self.assertEqual(rows0 | rows1, set(range(24)))

    def test_drop_last_epoch_shuffle_and_resume_seek_are_reproducible(self):
        prompts = _prompts(19)
        full, _ = _loader(prompts=prompts, tp_rank=1, dp_rank=0)
        full.set_epoch(3)
        expected = list(full)
        self.assertEqual(len(expected), 2)
        self.assertEqual(full.dataset_size, 4)

        resumed, _ = _loader(prompts=prompts, tp_rank=1, dp_rank=0)
        resumed.set_epoch(3)
        resumed.seek(1)
        self.assertEqual(
            [batch.sample_ids for batch in resumed],
            [batch.sample_ids for batch in expected[1:]],
        )
        with self.assertRaisesRegex(ValueError, "past the end"):
            resumed.seek(3)

        epoch0 = distributed_prompt_indices(19, dp_size=2, dp_rank=0, seed=19, epoch=0)
        epoch1 = distributed_prompt_indices(19, dp_size=2, dp_rank=0, seed=19, epoch=1)
        self.assertNotEqual(epoch0, epoch1)
        other_dp = distributed_prompt_indices(
            19, dp_size=2, dp_rank=1, seed=19, epoch=0
        )
        self.assertTrue(set(epoch0).isdisjoint(other_dp))
        self.assertEqual(len(epoch0) + len(other_dp), 18)

        too_small, _ = _loader(prompts=_prompts(3), tp_rank=0, dp_rank=0)
        self.assertEqual(too_small.dataset_size, 0)
        self.assertEqual(list(too_small), [])

    def test_padding_mode_matches_distributed_sampler(self):
        size = 19
        for rank in range(2):
            expected_sampler = torch.utils.data.DistributedSampler(
                list(range(size)),
                num_replicas=2,
                rank=rank,
                shuffle=True,
                seed=19,
                drop_last=False,
            )
            expected_sampler.set_epoch(3)
            actual = distributed_prompt_indices(
                size,
                dp_size=2,
                dp_rank=rank,
                seed=19,
                epoch=3,
                drop_last=False,
            )
            self.assertEqual(actual, list(expected_sampler))

        padded, _ = _loader(
            prompts=_prompts(size),
            tp_rank=0,
            dp_rank=0,
            sampler_drop_last=False,
        )
        self.assertEqual(padded.dataset_size, 4)

    def test_dspark_contract_includes_last_hidden_state_on_capture_device(self):
        loader, capture = _loader(
            prompts=_prompts(8), tp_rank=0, dp_rank=0, include_last=True
        )

        batch = next(iter(loader))

        self.assertEqual(
            set(batch.tensors),
            {
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
            },
        )
        devices = {tensor.device for tensor in batch.tensors.values()}
        self.assertEqual(devices, {capture.device})
        self.assertTrue(
            torch.equal(
                batch.tensors["target_last_hidden_states"],
                batch.tensors["hidden_states"] + 1000,
            )
        )

    def test_close_releases_in_process_capture(self):
        loader, capture = _loader(prompts=_prompts(8), tp_rank=0, dp_rank=0)
        loader.close()
        self.assertTrue(capture.closed)


if __name__ == "__main__":
    unittest.main()
