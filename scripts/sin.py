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


class Module(elegy.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def call(self, x):

        x = elegy.nn.Linear(64, name="backbone")(x)
        x = jax.nn.relu(x)

        y: np.ndarray = jnp.stack(
            [elegy.nn.Linear(2, name="expert")(x) for _ in range(self.k)], axis=1,
        )

        # y[..., 1] = 1.0 + jax.nn.elu(y[..., 1])
        y = jax.ops.index_update(y, jax.ops.index[..., 1], 1.0 + jax.nn.elu(y[..., 1]))

        logits = elegy.nn.Linear(self.k, name="gating")(x)
        probs = jax.nn.softmax(logits, axis=-1)

        return y, probs


def safe_log(x):
    return jnp.log(jnp.maximum(x, 1e-6))


class Loss(elegy.Loss):
    def __init__(self, original_version: bool = False):
        super().__init__()
        self.original_version = original_version

    def call(self, y_true, y_pred):
        y, probs = y_pred
        y_true = jnp.broadcast_to(y_true, (y_true.shape[0], y.shape[1]))

        if self.original_version:

            loss = -safe_log(
                jnp.sum(
                    probs
                    * jax.scipy.stats.norm.pdf(y_true, loc=y[..., 0], scale=y[..., 1]),
                    axis=1,
                ),
            )

            return loss
        else:
            components_loss = -jax.scipy.stats.norm.logpdf(
                y_true, loc=y[..., 0], scale=y[..., 1]
            )
            # components_loss = jnp.square(y_true - y[..., 0])

            mixture_loss = jnp.min(components_loss, axis=1)
            indexes = jnp.argmin(components_loss, axis=1)

            cce = elegy.losses.sparse_categorical_crossentropy(indexes, probs)

            elegy.add_metric(
                "acc", elegy.metrics.sparse_categorical_accuracy(indexes, probs)
            )

            return dict(assignments=cce, mixture=mixture_loss)


def show_single(X_train, y_train):
    m = LinearRegression()
    m.fit(X_train, y_train)
    x = np.linspace(X_train.min(), X_train.max(), 100)[..., None]
    y = m.predict(x)
    plt.scatter(X_train[..., 0], y_train[..., 0])
    plt.plot(x, y[:, 0], "k")

    plt.show()


def main(batch_size: int = 64, k: int = 3, debug: bool = False, original: bool = False):

    if debug:
        import debugpy

        debugpy.listen(5678)
        debugpy.wait_for_client()

    with plt.xkcd():
        # y_train = np.random.uniform(-np.pi, np.pi, size=(3000, 1))
        # X_train = np.sin(y_train) + np.random.normal(scale=0.1, size=y_train.shape)

        y_train = np.float32(np.random.uniform(-10.5, 10.5, (1, 3000))).T
        r_data = np.float32(np.random.normal(size=(3000, 1)))  # random noise
        X_train = np.float32(
            np.sin(0.75 * y_train) * 7.0 + y_train * 0.5 + r_data * 1.0
        )

        X_train = X_train / np.abs(X_train.max())
        y_train = y_train / np.abs(y_train.max())

        module = Module(k=k)
        model = elegy.Model(module, loss=Loss(original), optimizer=optix.adam(2e-3))

        model.summary(X_train[:batch_size], depth=1)

        show_single(X_train, y_train)

        if debug:
            model.run_eagerly = True

        x = np.linspace(X_train.min(), X_train.max(), 100)[..., None]
        for i in range(10):
            model.fit(
                x=X_train, y=y_train, epochs=100, batch_size=batch_size, shuffle=True
            )
            model.summary(X_train[:batch_size], depth=1)

            y, probs = model.predict(x)

            # with plt.xkcd():
            # plt.title(
            #     f"({float(probs[0][0]):.2f}) Normal({float(module.linear.w[0, 0]):.2f} x + {float(module.linear.b[0]):.2f}, {float(y[0, 0, 1]):.2f}), "
            #     f"({float(probs[0][1]):.2f}) Normal({float(module.linear_1.w[0, 0]):.2f} x + {float(module.linear_1.b[0]):.2f}, {float(y[0, 1, 1]):.2f}), "
            # )
            plt.figure()
            plt.scatter(X_train[..., 0], y_train[..., 0])

            for i in range(module.k):
                p = probs[:, i] > 0.02
                plt.plot(x[p], y[:, i, 0][p], "k")
                plt.plot(x[p], y[:, i, 0][p] + y[:, i, 1][p], "r")
                plt.plot(x[p], y[:, i, 0][p] - y[:, i, 1][p], "r")

            plt.figure()
            plt.title("P(z = k | x)")
            for i in range(module.k):
                sum_prev = probs[:, :i].sum(axis=-1)
                sum_current = probs[:, : i + 1].sum(axis=-1)
                plt.plot(x[..., 0], sum_current)
                plt.fill_between(x[..., 0], sum_current, sum_prev, alpha=0.30)

            plt.show()


if __name__ == "__main__":
    typer.run(main)
