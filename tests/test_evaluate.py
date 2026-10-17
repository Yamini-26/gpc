import jax
import jax.numpy as jnp
from flax import nnx

from gpc.architectures import DenoisingMLP
from gpc.envs import ParticleEnv
from gpc.evaluate import success_rate
from gpc.policy import Policy


def test_success_rate_always_succeeds() -> None:
    """If success_fn is always True, every rollout must count as a success."""
    env = ParticleEnv(episode_length=13)
    net = DenoisingMLP(
        action_size=env.task.model.nu,
        observation_size=env.observation_size,
        horizon=env.task.planning_horizon,
        hidden_layers=[32, 32],
        rngs=nnx.Rngs(0),
    )
    normalizer = nnx.BatchNorm(
        env.observation_size,
        momentum=0.1,
        epsilon=1e-5,
        use_bias=False,
        use_scale=False,
        rngs=nnx.Rngs(0),
    )
    policy = Policy(net, normalizer, env.task.u_min, env.task.u_max)

    rate = success_rate(
        env,
        policy,
        num_rollouts=4,
        warm_start_level=1.0,
        success_fn=lambda data: jnp.array(True),
        rng=jax.random.key(0),
        success_window=5,
    )
    assert rate == 1.0


def test_success_rate_always_fails() -> None:
    """If success_fn is always False, no rollout should count as a success."""
    env = ParticleEnv(episode_length=13)
    net = DenoisingMLP(
        action_size=env.task.model.nu,
        observation_size=env.observation_size,
        horizon=env.task.planning_horizon,
        hidden_layers=[32, 32],
        rngs=nnx.Rngs(0),
    )
    normalizer = nnx.BatchNorm(
        env.observation_size,
        momentum=0.1,
        epsilon=1e-5,
        use_bias=False,
        use_scale=False,
        rngs=nnx.Rngs(0),
    )
    policy = Policy(net, normalizer, env.task.u_min, env.task.u_max)

    rate = success_rate(
        env,
        policy,
        num_rollouts=4,
        warm_start_level=1.0,
        success_fn=lambda data: jnp.array(False),
        rng=jax.random.key(0),
        success_window=5,
    )
    assert rate == 0.0


if __name__ == "__main__":
    test_success_rate_always_succeeds()
    test_success_rate_always_fails()
