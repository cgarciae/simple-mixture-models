import typing as tp

import jax
import jax.numpy as jnp
import jax.ops
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import optix
from sklearn.linear_model import LinearRegression

import elegy

BATCH_SIZE = 64


X_train = np.random.uniform(-1, 1, size=(3000, 1))
y_train = np.concatenate(
    [
        0.5 + 0.3 * X_train[:1000] + np.random.normal(scale=0.1, size=(1000, 1)),
        0.4 + -0.4 * X_train[1000:] + np.random.normal(scale=0.05, size=(2000, 1)),
    ],
    axis=0,
)


m = LinearRegression()
m.fit(X_train, y_train)

x = np.linspace(-1.01, 1.01, 100)[..., None]
y = m.predict(x)


plt.scatter(X_train[..., 0], y_train[..., 0])
plt.plot(x, y[:, 0], "k")

plt.show()


class Module(elegy.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def call(self, x):
        batch_size = x.shape[0]

        y: np.ndarray = jnp.stack(
            [elegy.nn.Linear(2)(x) for _ in range(self.k)], axis=1
        )

        y = jax.ops.index_update(
            y, jax.ops.index[..., 1], 0.01 * (1.0 + jax.nn.elu(y[..., 1]))
        )

        logits = elegy.nn.Linear(self.k)(x)
        probs = jax.nn.softmax(logits, axis=-1)

        return y, probs


class Loss(elegy.Loss):
    def call(self, y_true, y_pred):

        y, probs = y_pred

        y_true = jnp.stack([y_true] * y.shape[1], axis=1)[..., 0]

        nll = 0.1 * -jax.scipy.stats.norm.logpdf(
            y_true, loc=y[..., 0], scale=y[..., 1]
        )  # + jnp.square(y_true - y[..., 0])

        min_nll = jnp.min(nll, axis=1)
        indexes = jnp.argmin(nll, axis=1)

        cce = elegy.losses.sparse_categorical_crossentropy(indexes, probs)

        return dict(cce=cce, nll=min_nll)


model = elegy.Model(Module(k=2), loss=Loss(), optimizer=optix.adam(1e-3))
module = model.module

# model.run_eagerly = True
x = np.linspace(-1.01, 1.01, 100)[..., None]
for i in range(5):
    model.fit(x=X_train, y=y_train, epochs=100, batch_size=BATCH_SIZE)

    y, probs = model.predict(x)

    # with plt.xkcd():
    plt.title(
        f"({float(probs[0][0]):.2f}) Normal({float(module.linear.w[0, 0]):.2f} x + {float(module.linear.b[0]):.2f}, {float(y[0, 0, 1]):.2f}), "
        f"({float(probs[0][1]):.2f}) Normal({float(module.linear_1.w[0, 0]):.2f} x + {float(module.linear_1.b[0]):.2f}, {float(y[0, 1, 1]):.2f}), "
    )
    plt.scatter(X_train[..., 0], y_train[..., 0])
    plt.plot(x, y[:, 0, 0], "k")
    plt.plot(x, y[:, 0, 0] + y[:, 0, 1], "r")
    plt.plot(x, y[:, 0, 0] - y[:, 0, 1], "r")
    plt.plot(x, y[:, 1, 0], "k")
    plt.plot(x, y[:, 1, 0] + y[:, 1, 1], "r")
    plt.plot(x, y[:, 1, 0] - y[:, 1, 1], "r")
    plt.show()
