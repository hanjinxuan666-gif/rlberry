from unittest.mock import Mock

import pytest

from rlberry.manager import evaluate_agents


@pytest.mark.parametrize("choose_random_agents", [False, True])
def test_evaluation_scores_stay_with_their_manager(choose_random_agents):
    managers = []
    for name, outputs in [("baseline", []), ("baseline", [2]), ("trained", [7])]:
        manager = Mock(agent_name=name, agent_handlers=[object()])
        manager.eval_agents.return_value = outputs
        managers.append(manager)

    scores = evaluate_agents(
        managers, n_simulations=1, choose_random_agents=choose_random_agents, plot=False
    )

    assert scores.to_dict("list") == {"baseline*": [2], "trained": [7]}
