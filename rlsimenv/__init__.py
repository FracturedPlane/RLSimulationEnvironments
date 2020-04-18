

### Utility functions

# from ..model.ModelUtil import clampAction
def clamp(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    """
    for i in range(len(actionV)):
        if actionV[i] < bounds[0][i]:
            actionV[i] = bounds[0][i]
        elif actionV[i] > bounds[1][i]:
            actionV[i] = bounds[1][i]
    return actionV 

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
    kwargs={'gui': False,
            "dt": 1/50.0}
)

gym_register(
    id='TagEnvPartiallyObserved-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 3.5,
            "grayscale": True,
            "fixed_view": False,
            "dt": 1/50.0
            }
)

gym_register(
    id='TagEnvPartiallyObserved-64x64-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 3.5,
            "observation_shape": (64, 64, 3),
            "observation_stack": 1,
            "flat_obs": False,
            "grayscale": False,
            "fixed_view": False,
            "dt": 1/50.0
            }
)
gym_register(
    id='TagEnvPartiallyObserved-1particle-64x64-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 6.5,
            "observation_shape": (64, 64, 3),
            "observation_stack": 1,
            "flat_obs": False,
            "grayscale": False,
            "fixed_view": False,
            "n_particles": 1,
            "dt": 1/50.0
            }
)

gym_register(
    id='TagEnvPartiallyObserved-1particle-flatobs-64x64-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 6.5,
            "observation_shape": (64, 64, 3),
            "observation_stack": 1,
            "flat_obs": True,
            "grayscale": False,
            "fixed_view": False,
            "n_particles": 1,
            "dt": 1/50.0
            }
)

gym_register(
    id='TagEnvPartiallyObserved-1particle-flatobs-16x16-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 6.5,
            "observation_shape": (16, 16, 3),
            "observation_stack": 1,
            "flat_obs": True,
            "grayscale": False,
            "fixed_view": False,
            "n_particles": 1,
            "dt": 1/50.0
            }
)

gym_register(
    id='TagEnvFullyObserved-1particle-flatobs-16x16-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 8.0,
            "observation_shape": (16, 16, 3),
            "observation_stack": 1,
            "flat_obs": True,
            "grayscale": False,
            "fixed_view": True,
            "n_particles": 1,
            "agent_scaling": 1.0,
            "dt": 1/50.0
            }
)

gym_register(
    id='TagEnvFullyObserved-1particle-flatobs-dualstate-16x16-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 8.0,
            "observation_shape": (16, 16, 3),
            "observation_stack": 1,
            "flat_obs": True,
            "grayscale": False,
            "fixed_view": True,
            "n_particles": 1,
            "agent_scaling": 1.0,
            "dt": 1/50.0,
            "include_true_state": True
            }
)

gym_register(
    id='TagEnvPartiallyObserved-1particle-flatobs-dualstate-16x16-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 6.5,
            "observation_shape": (16, 16, 3),
            "observation_stack": 1,
            "flat_obs": True,
            "grayscale": False,
            "fixed_view": False,
            "n_particles": 1,
            "agent_scaling": 1.0,
            "dt": 1/50.0,
            "include_true_state": True
            }
)

gym_register(
    id='TagEnvFullyObserved-1particle-flatobs-dualstate-16x16-render-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': True,
            "observation_height": 8.0,
            "observation_shape": (16, 16, 3),
            "observation_stack": 1,
            "flat_obs": True,
            "grayscale": False,
            "fixed_view": True,
            "n_particles": 1,
            "agent_scaling": 1.0,
            "dt": 1/50.0,
            "include_true_state": True
            }
)

gym_register(
    id='TagEnvPartiallyObserved-1particle-16x16-v0',
    entry_point='rlsimenv.TagEnv:TagEnv',
    reward_threshold=0.95,
    max_episode_steps=500,
    kwargs={'gui': False,
            "observation_height": 6.5,
            "observation_shape": (16, 16, 3),
            "observation_stack": 1,
            "flat_obs": False,
            "grayscale": False,
            "fixed_view": False,
            "n_particles": 1
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
