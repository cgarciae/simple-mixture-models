import typing as tp

import jax
import jax.numpy as jnp
import jax.ops
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax.experimental import optix
from sklearn.linear_model import LinearRegression

import elegy


class MixtureModel(elegy.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def call(self, x):

        x = elegy.nn.Linear(64, name="backbone")(x)
        x = jax.nn.relu(x)

        y: np.ndarray = jnp.stack(
            [elegy.nn.Linear(2, name="component")(x) for _ in range(self.k)], axis=1,
        )

        # equivalent to: y[..., 1] = 1.0 + jax.nn.elu(y[..., 1])
        y = jax.ops.index_update(y, jax.ops.index[..., 1], 1.0 + jax.nn.elu(y[..., 1]))

        logits = elegy.nn.Linear(self.k, name="gating")(x)
        probs = jax.nn.softmax(logits, axis=-1)

        return y, probs


class MixtureNLL(elegy.Loss):
    def call(self, y_true, y_pred):
        y, probs = y_pred
        y_true = jnp.broadcast_to(y_true, (y_true.shape[0], y.shape[1]))

        return -safe_log(
            jnp.sum(
                probs
                * jax.scipy.stats.norm.pdf(y_true, loc=y[..., 0], scale=y[..., 1]),
                axis=1,
            ),
        )


def main(batch_size: int = 64, k: int = 5, debug: bool = False):

    noise = np.float32(np.random.normal(size=(3000, 1)))  # random noise
    y_train = np.float32(np.random.uniform(-10.5, 10.5, (1, 3000))).T
    X_train = np.float32(np.sin(0.75 * y_train) * 7.0 + y_train * 0.5 + noise * 1.0)

    X_train = X_train / np.abs(X_train.max())
    y_train = y_train / np.abs(y_train.max())

    visualize_data(X_train, y_train)

    model = elegy.Model(
        module=MixtureModel(k=k), loss=MixtureNLL(), optimizer=optix.adam(3e-4)
    )

    model.summary(X_train[:batch_size], depth=1)

    model.fit(x=X_train, y=y_train, epochs=500, batch_size=batch_size, shuffle=True)

    visualize_model(X_train, y_train, model, k)


def visualize_model(X_train, y_train, model, k):

    x = np.linspace(X_train.min(), X_train.max(), 100)[..., None]
    y, probs = model.predict(x)

    plt.figure()
    plt.scatter(X_train[..., 0], y_train[..., 0])

    for i in range(k):
        p = probs[:, i] > 0.02
        plt.plot(x[p], y[:, i, 0][p], "k")
        plt.plot(x[p], y[:, i, 0][p] + y[:, i, 1][p], "r")
        plt.plot(x[p], y[:, i, 0][p] - y[:, i, 1][p], "r")

    plt.figure()
    plt.title("P(z = k | x)")
    for i in range(k):
        sum_prev = probs[:, :i].sum(axis=-1)
        sum_current = probs[:, : i + 1].sum(axis=-1)
        plt.plot(x[..., 0], sum_current)
        plt.fill_between(x[..., 0], sum_current, sum_prev, alpha=0.30)

    plt.show()


def safe_log(x):
    return jnp.log(jnp.maximum(x, 1e-6))


def visualize_data(X_train, y_train):
    m = LinearRegression()
    m.fit(X_train, y_train)
    x = np.linspace(X_train.min(), X_train.max(), 100)[..., None]
    y = m.predict(x)
    plt.scatter(X_train[..., 0], y_train[..., 0])
    plt.plot(x, y[:, 0], "k")

    plt.show()


if __name__ == "__main__":
    typer.run(main)
