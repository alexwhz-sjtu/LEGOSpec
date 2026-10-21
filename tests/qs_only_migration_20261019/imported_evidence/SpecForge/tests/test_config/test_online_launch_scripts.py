from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "train"


class OnlineLaunchScriptsTest(unittest.TestCase):
    def test_each_launcher_only_overrides_run_and_tracking_metadata(self):
        scripts = {
            "run_dflash.sh": (
                "qwen3-4b-dflash-colocated-online.yaml",
                "dflash-training",
            ),
            "run_dflash2.sh": (
                "qwen3-4b-dflash2-online.yaml",
                "dflash2-training",
            ),
            "run_dspark.sh": (
                "qwen3-4b-dspark-online.yaml",
                "dspark-training",
            ),
            "run_flashmtp.sh": (
                "qwen3-8b-flashmtp-online.yaml",
                "flashmtp_v3",
            ),
        }

        for filename, (recipe, project) in scripts.items():
            with self.subTest(script=filename):
                path = TRAIN_DIR / filename
                text = path.read_text()
                self.assertTrue(path.is_file())
                self.assertIn(recipe, text)
                self.assertIn(f"WANDB_PROJECT:-{project}", text)
                self.assertIn('ENABLE_WANDB="${ENABLE_WANDB:-true}"', text)
                self.assertIn('"run_id=$RUN_ID"', text)
                self.assertIn('"output_dir=$OUTPUT_DIR"', text)
                self.assertIn('"tracking.wandb_run_id=$WANDB_RUN_ID"', text)
                self.assertIn("clip_wandb_value", text)
                self.assertIsNone(
                    re.search(
                        r'"(?:model|data|training|deployment)\.[^"\s]+=.*"',
                        text,
                    )
                )

    def test_colocated_recipes_have_no_step_cap_or_run_metadata(self):
        recipes = (
            "qwen3-4b-dflash-colocated-online.yaml",
            "qwen3-4b-dflash2-online.yaml",
            "qwen3-4b-dspark-online.yaml",
            "qwen3-4b-flashmtp-online.yaml",
            "qwen3-8b-flashmtp-online.yaml",
        )
        config_dir = REPO_ROOT / "examples/configs/online/colocated"
        for recipe in recipes:
            with self.subTest(recipe=recipe):
                payload = yaml.safe_load((config_dir / recipe).read_text())
                self.assertNotIn("max_steps", payload["training"])
                self.assertNotIn("tracking", payload)
                self.assertNotIn("run_id", payload)
                self.assertNotIn("output_dir", payload)

    def test_flashmtp_launcher_derives_parameterized_run_name(self):
        text = (TRAIN_DIR / "run_flashmtp.sh").read_text()
        self.assertIn('"sf_flashmtp_"', text)
        for tag in (
            "draft_num_hidden_layers-",
            "flashmtp_markov_rank-",
            "flashmtp_chs_num_layers-",
            "flashmtp_local_position-",
            "flashmtp_backbone_conv_enabled-",
            "flashmtp_conv_kernel_size-",
            "flashmtp_conv_group_size-",
        ):
            with self.subTest(tag=tag):
                self.assertIn(tag, text)
        self.assertIn('OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_ID}"', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
