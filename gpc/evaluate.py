from typing import Callable

import jax
import jax.numpy as jnp
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.policy import Policy


def success_rate(
    env: TrainingEnv,
    policy: Policy,
    num_rollouts: int,
    warm_start_level: float,
    success_fn: Callable[[mjx.Data], jax.Array],
    rng: jax.Array,
    success_window: int = 10,
) -> jax.Array:
    """Run closed-loop policy rollouts and compute the fraction that succeed.

    Args:
        env: The training environment (defines reset/step/observation).
        policy: A trained GPC policy. No SPC is used here - the policy is
            the only controller.
        num_rollouts: Number of independent rollouts to evaluate.
        warm_start_level: Fixed alpha used to seed the policy's flow
            matching at every step.
        success_fn: Task-specific predicate, mjx.Data -> bool, e.g. "is the
            pole upright and slow".
        rng: Top-level RNG key.
        success_window: Number of final steps that must all satisfy
            success_fn for a rollout to count as a success.

    Returns:
        Fraction of rollouts (in [0, 1]) that were successful.
    """

    def _rollout(rng: jax.Array) -> jax.Array:
        rng, env_rng = jax.random.split(rng)
        x = env.init_state(env_rng)
        U = jnp.zeros((env.task.planning_horizon, env.task.model.nu))

        def _step(carry, _):
            x, U, rng = carry
            rng, policy_rng = jax.random.split(rng)
            y = env._get_observation(x)
            U = policy.apply(U, y, policy_rng, warm_start_level)
            u = U[0]
            x = env.step(x, u)
            return (x, U, rng), success_fn(x.data)

        (x, U, rng), successes = jax.lax.scan(
            _step, (x, U, rng), jnp.arange(env.episode_length)
        )

        # Succeed if stabilized over the final window, not just the last frame
        return jnp.all(successes[-success_window:])

    rngs = jax.random.split(rng, num_rollouts)
    successes = jax.vmap(_rollout)(rngs)
    return jnp.mean(successes)
