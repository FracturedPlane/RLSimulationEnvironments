import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering simpleworlds mujoco gym environments")

    """
    Reaching tasks
    """

    # register(
    #     id='SawyerMultitask-v0',
    #     entry_point='simpleworlds.envs.mujoco.sawyer_xyz'
    #                 '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
    #     tags={
    #         'git-commit-hash': 'ddd73dc',
    #         'author': 'murtaza',
    #     },
    #     kwargs=dict(
    #         goal_low=(-0.15, 0.4, 0.02, -.1, .45),
    #         goal_high=(0.15, 0.7, 0.02, .1, .65),
    #         puck_low=(-.1, .45),
    #         puck_high=(.1, .65),
    #         hand_low=(-0.15, 0.4, 0.02),
    #         hand_high=(0.15, .7, 0.02),
    #         norm_order=2,
    #         xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
    #         reward_type='state_distance',
    #         reset_free=False,
    #         clamp_puck_on_step=True,
    #     )
    # )


def create_image_48_sawyer_reach_xy_env_v1():
    from simpleworlds.core.image_env import ImageEnv
    from simpleworlds.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v1():
    from simpleworlds.core.image_env import ImageEnv
    from simpleworlds.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_v0():
    from simpleworlds.core.image_env import ImageEnv
    from simpleworlds.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_reset_free_v0():
    from simpleworlds.core.image_env import ImageEnv
    from simpleworlds.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaResetFreeEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

register_custom_envs()
