

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
    gym_register(
        id='ContinuousMaxwellsDemonWithGUI-v0',
        entry_point='rlsimenv.MaxwellsDemon:MaxwellsDemonEnvWithGUI',
        reward_threshold=0.95,
        max_episode_steps=500,
        obs_fov=60,
        observation_height=10,
        map_area=6,
        observation_shape=(64,64,3)
    )
    gym_register(
        id='ContinuousMaxwellsDemonFullyObserved-v0',
        entry_point='rlsimenv.MaxwellsDemon:MaxwellsDemonEnv',
        reward_threshold=0.95,
        max_episode_steps=500,
        render_shape=(128, 128, 3),
        observation_shape=(64, 64, 3),
        map_area=4,
        observation_height=15,        
    )
except:
    print ("gym not installed")
