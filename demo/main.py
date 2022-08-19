import argparse

import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache.compilation_cache import initialize_cache
from flax.jax_utils import replicate
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T

from .train import create_train_state, train_step, cross_replica_mean, test_step


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    device_count = jax.local_device_count()
    assert batch_size % device_count == 0, f'batch_size should be divisible by {device_count}'

    root = 'torchvision/datasets'
    specimen = jnp.empty((28, 28, 1))

    transform = T.Compose([
        T.ToTensor(),
        lambda X: torch.permute(X, (1, 2, 0)),  # (C, H, W) -> (H, W, C)
    ])

    train_dataset = MNIST(root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root, train=False, download=True, transform=transform)

    key = jax.random.PRNGKey(42)
    key_init, key = jax.random.split(key)
    num_classes = 10
    state = create_train_state(key_init, num_classes, learning_rate, specimen)
    state = replicate(state)


    print('===> Training')
    train_loader = DataLoader(train_dataset, batch_size)
    for epoch in range(epochs):
        state, loss = train_epoch(state, device_count, train_loader)
        with jnp.printoptions(precision=3):
            print(f'Epoch {epoch + 1}, train loss: {loss}')

    # Sync the batch statistics across replicas so that evaluation is deterministic.
    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))


    print('===> Testing')
    for dataset_name, dataset in [
            ('Train', train_dataset),
            ('Test', test_dataset)]:
        test_loader = DataLoader(dataset, batch_size)
        hits = test_epoch(state, device_count, test_loader)

        with jnp.printoptions(precision=3):
            total = len(dataset)
            accuracy = hits/total
            print(f'{dataset_name} accuracy: {accuracy}')


def train_epoch(state, device_count, loader):
    epoch_loss = 0
    for X, y in loader:
        remainder = X.shape[0] % device_count
        if remainder != 0:
            X = X[:-remainder]
            y = y[:-remainder]

        image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

        state, loss = train_step(state, image, label)
        epoch_loss += loss.sum()

    return state, epoch_loss


def test_epoch(state, device_count, test_loader):
    hits = 0
    for X, y in test_loader:
        remainder = X.shape[0] % device_count
        if remainder != 0:
            X = X[:-remainder]
            y = y[:-remainder]

        image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

        hits += test_step(state, image, label).sum()

    return hits


if __name__ == '__main__':
    initialize_cache('jit_cache')
    cli()
