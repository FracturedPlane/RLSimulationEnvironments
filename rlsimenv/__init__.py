

from rlsimenv.config import SIMULATION_ENVIRONMENTS


from gym.envs.registration import register as gym_register
# Use the gym_register because it allows us to set the max_episode_steps.
try:
    gym_register(
        id='ContinuousMaxwellsDemon-v0',
        entry_point='rlsimenv.MaxwellsDemon:MaxwellsDemonEnv',
        reward_threshold=0.95,
        max_episode_steps=500,
    )
except:
    print ("gym not installed")