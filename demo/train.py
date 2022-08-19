from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax

from .resnet import ResNet18


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]


def create_train_state(key: Any, num_classes: int, learning_rate: float, specimen: jnp.ndarray) -> TrainState:
    net = ResNet18(num_classes=num_classes)
    variables = net.init(key, specimen, True)
    tx = optax.adam(learning_rate)
    state = TrainState.create(
            apply_fn=net.apply,
            params=variables['params'],
            tx=tx,
            batch_stats=variables['batch_stats'],
    )
    return state


@partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
def train_step(state: TrainState, image: jnp.ndarray, label: jnp.ndarray) -> Tuple[TrainState, jnp.ndarray]:
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables, image, True, mutable=['batch_stats']
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, label)

        return loss.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)
    grads = jax.lax.psum(grads, axis_name='batch')

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    )

    return state, loss


cross_replica_mean: Callable = jax.pmap(lambda x: jax.lax.pmean(x, 'batch'), 'batch')

 
@jax.pmap
def test_step(state: TrainState, image: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, image, False)
    prediction = jnp.argmax(logits, axis=-1)
    hit = jnp.sum(prediction == label)

    return hit
