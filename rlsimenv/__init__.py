

from rlsimenv.config import SIMULATION_ENVIRONMENTS


from gym.envs.registration import register as gym_register
# Use the gym_register because it allows us to set the max_episode_steps.
# try:

print ("Registering rlsimenv environments")

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
    kwargs={'gui': False}
)

gym_register(
    id='MiniCraftBoxesFullyObservedGUI-v0',
    entry_point='rlsimenv.MiniCraftBoxes:MiniCraftBoxesFullyObserved',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': True}
)

gym_register(
    id='TagEnvFullyObserved-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False}
)

gym_register(
    id='TagEnvPartiallyObserved-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 3.5,
            "grayscale": True,
            "fixed_view": False
            }
)

gym_register(
    id='TagEnvPartiallyObserved-64x64-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 3.5,
            "grayscale": False,
            "fixed_view": False
            }
)

gym_register(
    id='TagEnvFullyObserved-64x64-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_shape": (64, 64, 3),
            "flat_obs": False,
            "observation_stack": 1,
            "grayscale": False}
)

gym_register(
    id='TagEnvFullyObservedGUI-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': True}
)
# except:
#     print ("gym not installed")
