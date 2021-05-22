from environment import create_env
from agent import create_agent

from agent_frame import gym_runner


if __name__ == "__main__":
    env = create_env()
    agent = create_agent()
    gym_runner.run(env, agent)
