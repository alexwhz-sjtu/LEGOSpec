from __future__ import annotations

import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config, migrate_legacy_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"


def _online_payload(
    *,
    backend: str = "sglang",
    modality: str = "text",
    topology: str = "disaggregated",
) -> dict:
    payload = {
        "model": {
            "target_model_path": "target",
            "draft_model_config": "draft.json",
            "vocab_mapping_path": "mapping.pt",
            "target_backend": backend,
            "input_modality": modality,
        },
        "data": {"train_data_path": "train.jsonl"},
        "training": {"strategy": "eagle3", "max_steps": 1},
        "deployment": {"mode": topology},
    }
    if topology == "disaggregated":
        payload["deployment"]["disaggregated"] = {
            "control_dir": "outputs/test/control",
            "capture_nnodes": 1,
            "backend": "mooncake",
            "server_urls": ["http://127.0.0.1:30000"],
            "mooncake_metadata_server": "http://127.0.0.1:35880/metadata",
            "mooncake_master_server_addr": "127.0.0.1:35551",
        }
    return payload


class ServerOnlyOnlineConfigTest(unittest.TestCase):
    def test_online_topology_matrix_and_local_parallel_constraints(self):
        local = _online_payload(topology="local_colocated")
        local["training"].update({"strategy": "dflash", "tp_size": 2})
        config = Config.model_validate(local)
        config.validate_world_size(8)
        with self.assertRaisesRegex(ValueError, "divisible"):
            config.validate_world_size(7)

        multi_node_local = _online_payload(topology="local_colocated")
        multi_node_local["training"]["strategy"] = "dflash"
        multi_node_local["deployment"]["trainer"] = {
            "nnodes": 2,
            "nproc_per_node": 8,
            "master_addr": "trainer-0",
        }
        with self.assertRaisesRegex(ValidationError, "nnodes=1"):
            Config.model_validate(multi_node_local)

        workers = _online_payload(topology="local_colocated")
        workers["training"]["strategy"] = "dflash"
        workers["data"]["dataloader_num_workers"] = 1
        with self.assertRaisesRegex(ValidationError, "dataloader_num_workers=0"):
            Config.model_validate(workers)

        sequence_parallel = _online_payload(topology="local_colocated")
        sequence_parallel["training"].update(
            {"strategy": "dflash", "sp_ulysses_size": 2}
        )
        with self.assertRaises(ValidationError):
            Config.model_validate(sequence_parallel)

        expert_parallel = _online_payload(topology="local_colocated")
        expert_parallel["training"].update({"strategy": "dflash", "tp_size": 2})
        expert_parallel["model"]["sglang_ep_size"] = 3
        with self.assertRaisesRegex(ValidationError, "sglang_ep_size"):
            Config.model_validate(expert_parallel)

        external = _online_payload(topology="disaggregated")
        self.assertEqual(
            Config.model_validate(external).deployment.disaggregated.capture_nnodes,
            1,
        )
        external["deployment"]["disaggregated"].pop("capture_nnodes")
        with self.assertRaisesRegex(ValidationError, "capture_nnodes"):
            Config.model_validate(external)

    def test_removed_managed_local_config_has_migration_error(self):
        payload = _online_payload(topology="disaggregated")
        payload["deployment"]["disaggregated"]["managed_local"] = {}
        with self.assertRaisesRegex(ValueError, "managed_local was removed"):
            migrate_legacy_config(payload)

    def test_colocated_online_support_is_algorithm_gated(self):
        payload = _online_payload(topology="local_colocated")
        config = Config.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "currently supports dflash"):
            resolve_run(config)
        payload["training"]["strategy"] = "dflash"
        self.assertEqual(
            resolve_run(Config.model_validate(payload)).algorithm.name,
            "dflash",
        )

    def test_online_requires_the_sglang_server_backend(self):
        # The retired in-process backends are no longer schema values at all,
        # so the rejection is the enum itself rather than the online validator.
        for backend in ("hf", "custom"):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValidationError, "target_backend"),
            ):
                Config.model_validate(_online_payload(backend=backend))

    def test_vlm_is_explicitly_unsupported(self):
        config = Config.model_validate(_online_payload(modality="qwen2_5_vl"))
        self.assertEqual(config.model.input_modality, "qwen2_5_vl")
        with self.assertRaisesRegex(
            ValueError,
            "no streaming feature contract and provider for modality " "'qwen2_5_vl'",
        ):
            resolve_run(config)

    def test_offline_configs_reject_retired_backends_instead_of_ignoring_them(self):
        # Offline consumers never instantiate a target inference backend, but a
        # config naming a retired backend must fail at load rather than be
        # silently ignored — training.md promises retired options stay loud.
        def _offline_payload(backend):
            return {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                    "vocab_mapping_path": "mapping.pt",
                    "target_backend": backend,
                },
                "data": {"hidden_states_path": "features"},
            }

        config = Config.model_validate(_offline_payload("sglang"))
        self.assertEqual(config.mode, "offline")
        for backend in ("hf", "custom"):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValidationError, "target_backend"),
            ):
                Config.model_validate(_offline_payload(backend))

    def test_every_online_recipe_uses_a_supported_data_plane(self):
        online_recipes = []
        for path in sorted(EXAMPLE_CONFIG_DIR.rglob("*.yaml")):
            payload = yaml.safe_load(path.read_text())
            data = payload["data"]
            if not (data.get("train_data_path") or data.get("prompts_path")):
                continue
            online_recipes.append(path.name)
            self.assertEqual(payload["model"]["target_backend"], "sglang")
            self.assertFalse(payload["model"].get("shard_target_output", False))
            if payload["deployment"]["mode"] == "disaggregated":
                self.assertEqual(
                    payload["deployment"]["disaggregated"]["backend"], "mooncake"
                )
                self.assertGreater(
                    payload["deployment"]["disaggregated"]["capture_nnodes"], 0
                )
            else:
                self.assertEqual(payload["deployment"]["mode"], "local_colocated")
                self.assertNotIn("disaggregated", payload["deployment"])
        self.assertTrue(online_recipes)
        self.assertFalse(
            any(EXAMPLE_CONFIG_DIR.rglob("qwen2.5-vl-7b-eagle3-online.yaml"))
        )
        self.assertFalse(
            any(EXAMPLE_CONFIG_DIR.rglob("qwen2.5-vl-32b-eagle3-online.yaml"))
        )

    def test_application_resolution_accepts_the_server_only_contract(self):
        resolved = resolve_run(Config.model_validate(_online_payload()))
        self.assertEqual(resolved.algorithm.name, "eagle3")
        self.assertEqual(resolved.config.deployment.mode, "disaggregated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
