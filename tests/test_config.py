from unittest import TestCase

from core.budget import BudgetLimits
from core.config import AppConfig, ConfigError, load_config


class LoadConfigTests(TestCase):
    def test_defaults_are_complete_and_valid(self):
        config = load_config(env={})
        self.assertEqual(config.model_mode, "hybrid")
        self.assertEqual(config.local_model, "system")
        self.assertEqual(config.pcc_model, "pcc")
        self.assertEqual(config.local_context_token_budget, 3000)
        self.assertEqual(config.pcc_context_token_budget, 12000)
        self.assertEqual(config.compact_trigger_ratio, 0.8)
        self.assertIsInstance(config.budget, BudgetLimits)

    def test_model_mode_aliases(self):
        self.assertEqual(load_config(env={"MODEL_MODE": "auto"}).model_mode, "hybrid")
        self.assertEqual(load_config(env={"MODEL_MODE": "server"}).model_mode, "pcc")
        self.assertEqual(load_config(env={"MODEL_MODE": "local"}).model_mode, "local")

    def test_invalid_model_mode_is_reported(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"MODEL_MODE": "banana"})
        self.assertTrue(any("MODEL_MODE" in problem for problem in caught.exception.problems))

    def test_cloud_mode_requires_credentials(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"MODEL_MODE": "cloud"})
        for env_name in ("CLOUD_BASE_URL", "CLOUD_API_KEY", "CLOUD_MODEL"):
            self.assertTrue(
                any(env_name in problem for problem in caught.exception.problems),
                env_name,
            )

    def test_cloud_mode_valid_config(self):
        config = load_config(
            env={
                "MODEL_MODE": "cloud",
                "CLOUD_BASE_URL": "https://example.test/v1",
                "CLOUD_API_KEY": "sk-test",
                "CLOUD_MODEL": "qwen-plus",
            }
        )
        self.assertEqual(config.model_mode, "cloud")
        self.assertEqual(config.cloud_model, "qwen-plus")
        self.assertEqual(config.cloud_prompt_profile.value, "full")
        self.assertEqual(config.cloud_context_token_budget, 12000)

    def test_cloud_prompt_profile_and_budget_overrides(self):
        config = load_config(
            env={
                "MODEL_MODE": "cloud",
                "CLOUD_BASE_URL": "https://example.test/v1",
                "CLOUD_API_KEY": "sk-test",
                "CLOUD_MODEL": "qwen-plus",
                "CLOUD_PROMPT_PROFILE": "mini",
                "CLOUD_CONTEXT_TOKEN_BUDGET": "8000",
            }
        )
        self.assertEqual(config.cloud_prompt_profile.value, "mini")
        self.assertEqual(config.cloud_context_token_budget, 8000)

    def test_cloud_budget_uses_shared_fallback(self):
        config = load_config(env={"CONTEXT_TOKEN_BUDGET": "4000"})
        self.assertEqual(config.cloud_context_token_budget, 4000)

    def test_stt_defaults_fall_back_to_cloud_credentials(self):
        config = load_config(
            env={
                "CLOUD_BASE_URL": "https://api.groq.com/openai/v1",
                "CLOUD_API_KEY": "gsk-test",
            }
        )
        self.assertEqual(config.stt_base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(config.stt_api_key, "gsk-test")
        self.assertEqual(config.stt_model, "whisper-large-v3-turbo")

    def test_stt_overrides_take_precedence(self):
        config = load_config(
            env={
                "CLOUD_BASE_URL": "https://api.groq.com/openai/v1",
                "CLOUD_API_KEY": "gsk-test",
                "STT_BASE_URL": "https://stt.example.test/v1",
                "STT_API_KEY": "stt-key",
                "STT_MODEL": "whisper-large-v3",
            }
        )
        self.assertEqual(config.stt_base_url, "https://stt.example.test/v1")
        self.assertEqual(config.stt_api_key, "stt-key")
        self.assertEqual(config.stt_model, "whisper-large-v3")

    def test_stt_is_disabled_without_credentials(self):
        config = load_config(env={})
        self.assertEqual(config.stt_api_key, "")
        self.assertEqual(config.stt_base_url, "")

    def test_notes_file_default_and_override(self):
        self.assertEqual(load_config(env={}).notes_file, "notes.json")
        self.assertEqual(load_config(env={"NOTES_FILE": "my.json"}).notes_file, "my.json")

    def test_invalid_number_is_reported(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"LOCAL_CONTEXT_TOKEN_BUDGET": "много"})
        self.assertTrue(
            any("LOCAL_CONTEXT_TOKEN_BUDGET" in problem for problem in caught.exception.problems)
        )

    def test_out_of_range_values_are_reported(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"COMPACT_TRIGGER_RATIO": "1.5"})
        self.assertTrue(caught.exception.problems)

    def test_context_token_fallback_variable(self):
        config = load_config(env={"CONTEXT_TOKEN_BUDGET": "4000"})
        self.assertEqual(config.local_context_token_budget, 4000)
        self.assertEqual(config.pcc_context_token_budget, 4000)

    def test_explicit_budgets_override_fallback(self):
        config = load_config(
            env={"CONTEXT_TOKEN_BUDGET": "4000", "PCC_CONTEXT_TOKEN_BUDGET": "9000"}
        )
        self.assertEqual(config.local_context_token_budget, 4000)
        self.assertEqual(config.pcc_context_token_budget, 9000)

    def test_unknown_prompt_profile_is_reported(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"PROMPT_PROFILE": "mega"})
        self.assertTrue(any("профиль" in problem for problem in caught.exception.problems))

    def test_prompt_profile_override_applies_to_both(self):
        config = load_config(env={"PROMPT_PROFILE": "full"})
        self.assertEqual(config.local_prompt_profile.value, "full")
        self.assertEqual(config.pcc_prompt_profile.value, "full")

    def test_telegram_required_for_telegram_interface(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={}, telegram_required=True)
        self.assertTrue(
            any("TELEGRAM_BOT_TOKEN" in problem for problem in caught.exception.problems)
        )

    def test_allowed_user_id_required_with_token(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"TELEGRAM_BOT_TOKEN": "123:abc"})
        self.assertTrue(
            any("ALLOWED_USER_ID" in problem for problem in caught.exception.problems)
        )

    def test_telegram_config_valid_with_both_values(self):
        config = load_config(
            env={"TELEGRAM_BOT_TOKEN": "123:abc", "ALLOWED_USER_ID": "42"},
            telegram_required=True,
        )
        self.assertEqual(config.allowed_user_id, "42")

    def test_hybrid_requires_distinct_model_aliases(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"LOCAL_MODEL": "same", "PCC_MODEL": "same"})
        self.assertTrue(
            any("различаться" in problem for problem in caught.exception.problems)
        )

    def test_same_aliases_allowed_in_strict_modes(self):
        config = load_config(env={"MODEL_MODE": "local", "LOCAL_MODEL": "x", "PCC_MODEL": "x"})
        self.assertEqual(config.model_mode, "local")

    def test_run_budget_env_overrides(self):
        config = load_config(
            env={
                "RUN_MAX_STEPS": "4",
                "RUN_MAX_SECONDS": "60",
                "RUN_MAX_IDENTICAL_CALLS": "1",
            }
        )
        self.assertEqual(config.budget.max_steps, 4)
        self.assertEqual(config.budget.max_wall_seconds, 60.0)
        self.assertEqual(config.budget.max_identical_calls, 1)
        self.assertEqual(config.budget.max_model_calls, 12)

    def test_invalid_run_budget_is_reported(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"RUN_MAX_STEPS": "ноль"})
        self.assertTrue(any("RUN_MAX_STEPS" in problem for problem in caught.exception.problems))

    def test_zero_budget_is_reported(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"RUN_MAX_STEPS": "0"})
        self.assertTrue(caught.exception.problems)

    def test_bash_workspace_and_timeout(self):
        config = load_config(env={"BASH_WORKSPACE": "~/agent-ws", "BASH_TIMEOUT": "45"})
        self.assertTrue(str(config.bash_workspace).endswith("agent-ws"))
        self.assertEqual(config.bash_timeout, 45.0)

    def test_bash_approval_validation(self):
        self.assertEqual(load_config(env={"BASH_APPROVAL": "prompt"}).bash_approval.value, "prompt")
        with self.assertRaises(ConfigError):
            load_config(env={"BASH_APPROVAL": "always"})

    def test_force_depth_validation(self):
        self.assertEqual(load_config(env={"WEB_SEARCH_FORCE_DEPTH": "deep"}).web_search_force_depth.value, "deep")
        with self.assertRaises(ConfigError):
            load_config(env={"WEB_SEARCH_FORCE_DEPTH": "turbo"})

    def test_multiple_problems_reported_together(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(
                env={
                    "MODEL_MODE": "banana",
                    "COMPACT_TRIGGER_RATIO": "2",
                    "RUN_MAX_STEPS": "-1",
                }
            )
        self.assertGreaterEqual(len(caught.exception.problems), 3)

    def test_config_error_message_lists_problems(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(env={"MODEL_MODE": "banana"})
        self.assertIn("MODEL_MODE", str(caught.exception))

    def test_app_config_direct_construction(self):
        config = AppConfig()
        self.assertEqual(config.budget.max_steps, 8)


if __name__ == "__main__":
    import unittest

    unittest.main()
