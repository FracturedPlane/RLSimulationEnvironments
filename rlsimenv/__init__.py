

from rlsimenv.config import SIMULATION_ENVIRONMENTS


from gym.envs.registration import register as gym_register
# Use the gym_register because it allows us to set the max_episode_steps.
# try:

gym_register(
    id='ContinuousMaxwellsDemon-v0',
    entry_point='rlsimenv.MaxwellsDemon:MaxwellsDemonPartiallyObserved',
    reward_threshold=0.95,
    max_episode_steps=500,
)
gym_register(
    id='ContinuousMaxwellsDemonFullyObserved-v0',
    entry_point='rlsimenv.MaxwellsDemon:MaxwellsDemonFullyObserved',
    reward_threshold=0.95,
    max_episode_steps=500,
)

gym_register(
    id='MiniCraftBoxesFullyObserved-v0',
    entry_point='rlsimenv.MiniCraftBoxes:MiniCraftBoxesFullyObserved',
    reward_threshold=0.95,
    max_episode_steps=500,
)

gym_register(
    id='MiniCraftBoxesFullyObservedGUI-v0',
    entry_point='rlsimenv.MiniCraftBoxes:MiniCraftBoxesFullyObserved',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': True}
)
# except:
#     print ("gym not installed")
