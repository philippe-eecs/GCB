import torch
from goalbisim.agents.goalpixelsac import GoalPixelSACAgent
from goalbisim.agents.goalpixeliql import GoalPixelIQLAgent
from goalbisim.agents.goalpixelbc import GoalPixelBCAgent


def agent_initalization(details, env, device, eval_transforms, actor_representation, critic_representation, target_critic_representation):

    rl_algorithm = details['rl_algorithm']

    if rl_algorithm == 'SAC':
        agent = GoalPixelSACAgent(env.observation_space.shape, env.action_space.shape, device, eval_transforms, actor_representation, critic_representation, target_critic_representation,
        None, **details['sac_kwargs'])

    elif rl_algorithm == 'IQL':
        agent = GoalPixelIQLAgent(env.observation_space.shape, env.action_space.shape, device, eval_transforms, actor_representation, critic_representation, target_critic_representation,
        None, **details['iql_kwargs'])

    elif rl_algorithm == 'BC':
        agent = GoalPixelBCAgent(env.observation_space.shape, env.action_space.shape, device, eval_transforms, actor_representation, None, **details['bc_kwargs'])

    else:
        raise NotImplementedError

    return agent

