from unittest import TestCase

from core.agent import Agent
from core.model_router import ModelRoute, RouteDecision


class AgentRoutingTests(TestCase):
    def test_route_changes_model_prompt_and_budget_before_turn(self):
        route = ModelRoute("pcc", "pcc", "FULL", 12000, "system")
        agent = Agent(None, "system", "MINI", token_budget=3000)
        agent.route_selector = lambda _: RouteDecision(route, "complex task", 2)

        agent._select_route("реализуй функцию")

        self.assertEqual(agent.model, "pcc")
        self.assertEqual(agent.base_system, "FULL")
        self.assertEqual(agent.token_budget, 12000)
        self.assertEqual(agent.model_fallback, "system")
        self.assertEqual(agent.last_route_name, "pcc")
        self.assertEqual(agent.messages[0]["content"], "FULL")
