# noinspection PyPep8Naming
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer


# check all but first shape index
def safe_mult(a, b):
    if len(a.shape) < 2:
        raise ValueError("Expected shape to be at least rank 2")
    if str(a.shape[0]) != "?":
        raise ValueError("Expected first shape entry to be ?")
    ra = len(a.shape)
    if ra != len(b.shape):
        raise ValueError("input ranks did not match")
    if str(b.shape[0]) != "?":
        raise ValueError("Expected first shape entry to be ?")
    for i in range(1, ra):
        if a.shape[i] != b.shape[i]:
            raise ValueError("input shapes did not match")
    c = a * b
    if len(c.shape) != ra:
        raise ValueError("result rank did not match")
    for i in range(1, ra):
        if a.shape[i] != c.shape[i]:
            raise ValueError("result shape did not match")
    if str(c.shape[0]) != "?":
        raise ValueError("Expected first shape entry to be ?")
    return c


class ScoringLayer(Layer):
    def __init__(self, *, 
                 alpha=0.1, 
                 debug=False,
                 var_ratio = True):
        self.alpha = alpha
        self.debug = debug
        self.var_ratio = var_ratio
        self.var_ratio_smoothing = 1.e-2
        super(ScoringLayer, self).__init__()

    def build(self, input_shape):
        super(ScoringLayer, self).build(input_shape)

    def compute_variational_loss(self, *, x, y_true, msg_id=0):
        var_loss = None
        if self.alpha <= 0:
            return var_loss, msg_id
        n_internal_layers = len(x) - 2
        if n_internal_layers <= 0:
            return var_loss, msg_id
        layers_normalization = n_internal_layers * (n_internal_layers + 1) / 2
        y_triples = [(y_is, row_indicator, K.sum(row_indicator) + 1.0e-6) for
                     y_is, row_indicator in [(1, y_true), (0, 1 - y_true)]]  # assuming y_true is 0/1
        for y_is, row_indicator, row_weight in y_triples:
            if len(row_weight.shape) != 0:
                raise ValueError("Expected row_weight.shape to be 0")
        for i in range(1, len(x) - 1):  # all but first and last layer
            layer = x[i]
            layer_weight = i / (layers_normalization * layer.shape.as_list()[1])
            for j in range(layer.shape[1]):
                xij = layer[:, j:(j + 1)]  # try to keep shape
                # y-pass 1/2 get conditional distributions and means
                y_derived = dict()
                for y_is, row_indicator, row_weight in y_triples:
                    coords = (
                            "(" + "y=" + str(y_is) + ", i=" + str(i) + ", j=" + str(j) + ")"
                    )
                    if self.debug:
                        xij = K.print_tensor(
                            xij, message=str(msg_id).zfill(3) + " " + "xij" + coords
                        )
                        msg_id = msg_id + 1
                    xij_conditional = safe_mult(row_indicator, xij)
                    if self.debug:
                        xij_conditional = K.print_tensor(
                            xij_conditional,
                            message=str(msg_id).zfill(3)
                            + " "
                            + "xij_conditional"
                            + coords,
                        )
                        msg_id = msg_id + 1
                    xbar = K.sum(xij_conditional) / row_weight
                    if self.debug:
                        xbar = K.print_tensor(
                            xbar, message=str(msg_id).zfill(3) + " " + "xbar" + coords
                        )
                        msg_id = msg_id + 1
                    y_derived[y_is] = (xij_conditional, xbar)
                mean_sq_diff = 1
                if self.var_ratio:
                    xbar_0 = y_derived[0][1]
                    if len(xbar_0.shape) != 0:
                        raise ValueError("Expected xbar_0.shape to be 0")
                    xbar_1 = y_derived[1][1]
                    if len(xbar_1.shape) != 0:
                        raise ValueError("Expected xbar_1.shape to be 0")
                    mean_sq_diff = (xbar_1 - xbar_0)**2 + self.var_ratio_smoothing
                    if len(mean_sq_diff.shape) != 0:
                        raise ValueError("Expected mean_sq_diff.shape to be 0")
                    if self.debug:
                        coords = (
                                "(" + "i=" + str(i) + ", j=" + str(j) + ")"
                        )
                        mean_sq_diff = K.print_tensor(
                            mean_sq_diff,
                            message=str(msg_id).zfill(3) + " " + "mean_sq_diff" + coords,
                        )
                        msg_id = msg_id + 1
                # y-pass 2/2 compute conditional variances
                for y_is, row_indicator, row_weight in y_triples:
                    coords = (
                            "(" + "y=" + str(y_is) + ", i=" + str(i) + ", j=" + str(j) + ")"
                    )
                    xij_conditional, xbar = y_derived[y_is]
                    if len(xbar.shape) != 0:
                        raise ValueError("Expected xbar.shape to be 0")
                    diff_ij = xij - xbar
                    if self.debug:
                        diff_ij = K.print_tensor(
                            diff_ij,
                            message=str(msg_id).zfill(3) + " " + "diff_ij" + coords,
                        )
                        msg_id = msg_id + 1
                    diff_ij_conditional = safe_mult(row_indicator, diff_ij)
                    if self.debug:
                        diff_ij_conditional = K.print_tensor(
                            diff_ij_conditional,
                            message=str(msg_id).zfill(3)
                            + " "
                            + "diff_ij_conditional"
                            + coords,
                        )
                        msg_id = msg_id + 1
                    # ratio of y-conditioned var over y-different var
                    conditional_var = diff_ij_conditional**2 / mean_sq_diff
                    wij = self.alpha * layer_weight
                    if self.debug:
                        conditional_var = K.print_tensor(
                            conditional_var,
                            message=str(msg_id).zfill(3)
                            + " "
                            + "conditional_var"
                            + coords
                            + " * "
                            + str(wij),
                        )
                        msg_id = msg_id + 1
                    if var_loss is None:
                        var_loss = wij * conditional_var
                    else:
                        var_loss = var_loss + wij * conditional_var
        return var_loss, msg_id

    def call(self, x, **kwargs):
        if not isinstance(x, list):
            raise TypeError("Expected x to be a list")
        msg_id = 0
        first_item = x[0]
        y_true = first_item[
            :, (first_item.shape[1] - 1):(first_item.shape[1])
        ]  # keep shape
        if self.debug:
            y_true = K.print_tensor(
                y_true, message=str(msg_id).zfill(3) + " " + "y_true"
            )
            msg_id = msg_id + 1
        last_item = x[len(x) - 1]
        y_pred = last_item
        # per-row cross-entropy or deviance/2 part of loss
        eps = 1.0e-6
        y_pred = K.maximum(y_pred, eps)
        y_pred = K.minimum(y_pred, 1 - eps)
        if self.debug:
            y_pred = K.print_tensor(
                y_pred, message=str(msg_id).zfill(3) + " " + "y_pred"
            )
            msg_id = msg_id + 1
        loss = -safe_mult(y_true, K.log(y_pred)) - safe_mult(1 - y_true, K.log(1 - y_pred))
        if self.debug:
            loss = K.print_tensor(
                loss, message=str(msg_id).zfill(3) + " " + "entropy loss"
            )
            msg_id = msg_id + 1
        # conditional clustered action/variation on activation
        var_loss, msg_id = self.compute_variational_loss(
            x=x, y_true=y_true, msg_id=msg_id
        )
        if var_loss is not None:
            if self.debug:
                var_loss = K.print_tensor(
                    var_loss, message=str(msg_id).zfill(3) + " " + "variational loss"
                )
                msg_id = msg_id + 1
            loss = loss + var_loss
        if self.debug:
            loss = K.print_tensor(
                loss, message=str(msg_id).zfill(3) + " " + "final squared loss"
            )
            msg_id = msg_id + 1
        loss = K.sqrt(loss)
        if self.debug:
            loss = K.print_tensor(
                loss, message=str(msg_id).zfill(3) + " " + "final loss"
            )
            # noinspection PyUnusedLocal
            msg_id = msg_id + 1
        return loss

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise TypeError("Expected x to be a list")
        last_shape = input_shape[len(input_shape) - 1]
        return last_shape
