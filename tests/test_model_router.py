from unittest import TestCase

from core.model_router import AppleModelRouter, ModelRoute


class AppleModelRouterTests(TestCase):
    def setUp(self):
        self.local = ModelRoute("local", "system", "MINI", 3000, "pcc")
        self.pcc = ModelRoute("pcc", "pcc", "FULL", 12000, "system")

    def test_simple_request_stays_local(self):
        decision = AppleModelRouter(self.local, self.pcc).select("Привет, как дела?")

        self.assertEqual(decision.route, self.local)
        self.assertEqual(decision.reason, "simple request")

    def test_complex_development_request_uses_pcc(self):
        decision = AppleModelRouter(self.local, self.pcc).select(
            "Проанализируй логи и исправь архитектуру агента"
        )

        self.assertEqual(decision.route, self.pcc)
        self.assertGreaterEqual(decision.score, 2)

    def test_code_block_uses_pcc(self):
        decision = AppleModelRouter(self.local, self.pcc).select("Что не так?\n```py\n1/0\n```")

        self.assertEqual(decision.route, self.pcc)

    def test_short_follow_up_keeps_pcc_for_same_task(self):
        router = AppleModelRouter(self.local, self.pcc)
        router.select("Добавь команду в проект и протестируй")

        decision = router.select("да")

        self.assertEqual(decision.route, self.pcc)
        self.assertEqual(decision.reason, "follow-up to complex task")

    def test_new_simple_question_returns_to_local(self):
        router = AppleModelRouter(self.local, self.pcc)
        router.select("Проанализируй архитектуру проекта")

        decision = router.select("Который час?")

        self.assertEqual(decision.route, self.local)

    def test_forced_modes_override_heuristics(self):
        complex_request = "Реализуй и протестируй сложную архитектуру"
        local_route = AppleModelRouter(
            self.local, self.pcc, "local"
        ).select(complex_request).route
        self.assertEqual(local_route.model, "system")
        self.assertIsNone(local_route.fallback_model)
        self.assertEqual(
            AppleModelRouter(self.local, self.pcc, "pcc").select("Привет").route,
            self.pcc,
        )
