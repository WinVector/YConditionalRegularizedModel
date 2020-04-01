import numpy

from tensorflow.keras.models import Model
from tensorflow.keras import Input

from TrimmingLayer import TrimmingLayer
from ScoringLayer import ScoringLayer


class YConditionalRegularizedModel:
    def __init__(
        self,
        model_steps_factory,
        *,
        alpha=1.0e-3,
        epochs=100,
        batch_size=100,
        verbose=0,
        debug=False
    ):
        self.model_steps_factory = model_steps_factory
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.debug = debug
        self.scoring_model = None

    @staticmethod
    def create_model_loss_adapted(steps, *, alpha, debug):
        input_layer = Input(shape=(steps[0] + 1,))
        collected_steps = [input_layer]
        current_step = TrimmingLayer()(
            input_layer
        )  # hide last/output column from most layers
        for i in range(1, len(steps)):
            current_step = steps[i](current_step)
            collected_steps.append(current_step)
        cnl = ScoringLayer(alpha=alpha, debug=debug)(collected_steps)
        new_model = Model(inputs=[input_layer], outputs=[cnl])
        # Compile model
        new_model.compile(loss="mean_squared_error", optimizer="adam")
        return new_model

    @staticmethod
    def create_model_direct(steps):
        input_layer = Input(shape=(steps[0],))
        current_step = input_layer
        for i in range(1, len(steps)):
            current_step = steps[i](current_step)
        new_model = Model(inputs=[input_layer], outputs=[current_step])
        # Compile model
        new_model.compile(loss="mean_squared_error", optimizer="adam")
        return new_model

    # noinspection PyPep8Naming
    def fit(self, X, y):
        # add outcome as last column
        X_with_outcome = numpy.concatenate((X, y.reshape((y.shape[0], 1))), 1)
        loss_model = YConditionalRegularizedModel.create_model_loss_adapted(
            self.model_steps_factory(), alpha=self.alpha, debug=self.debug
        )
        self.scoring_model = YConditionalRegularizedModel.create_model_direct(
            self.model_steps_factory()
        )
        loss_model.fit(
            X_with_outcome,
            numpy.zeros(X_with_outcome.shape[0]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self.scoring_model.set_weights(loss_model.get_weights())

    # noinspection PyPep8Naming
    def predict_proba(self, X):
        n = X.shape[0]
        probs = self.scoring_model.predict(X)
        pm1 = 1 - probs
        preds = numpy.concatenate((pm1.reshape((n, 1)), probs.reshape((n, 1))), 1)
        return preds

    # noinspection PyPep8Naming
    def fit_predict_proba(self, X, y):
        self.fit(X, y)
        return self.predict_proba(X)

    # noinspection PyPep8Naming
    def predict(self, X):
        probs = self.predict_proba(X)
        return probs[:, 1] >= 0.5

    # noinspection PyPep8Naming
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    # noinspection PyPep8Naming
    def transform(self, X):
        return self.predict_proba(X)

    # noinspection PyPep8Naming
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
