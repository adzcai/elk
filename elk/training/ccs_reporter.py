"""An ELK reporter network."""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, cast
from jaxtyping import Float
from einops import rearrange

import torch
import torch.nn as nn
import warnings
from torch import Tensor
from torch.nn.functional import binary_cross_entropy as bce

from ..metrics import roc_auc
from ..parsing import parse_loss
from ..utils.typing import assert_type
from .classifier import Classifier
from .losses import LOSSES, LogitsMultiChoice
from .normalizer import NormalizationMode, Normalizer
from .reporter import Reporter, ReporterConfig


Hiddens = Float[Tensor, "batch n_variants num_classes hidden_size"]


@dataclass
class CcsReporterConfig(ReporterConfig):
    """
    Args:
        activation: The activation function to use. Defaults to GELU.
        bias: Whether to use a bias term in the linear layers. Defaults to True.
        hidden_size: The number of hidden units in the MLP. Defaults to None.
            By default, use an MLP expansion ratio of 4/3. This ratio is used by
            Tucker et al. (2022) <https://arxiv.org/abs/2204.09722> in their 3-layer
            MLP probes. We could also use a ratio of 4, imitating transformer FFNs,
            but this seems to lead to excessively large MLPs when num_layers > 2.
        init: The initialization scheme to use. Defaults to "zero".
        loss: The loss function to use. list of strings, each of the form
            "coef*name", where coef is a float and name is one of the keys in
            `elk.training.losses.LOSSES`.
            Example: --loss 1.0*consistency_squared 0.5*prompt_var
            corresponds to the loss function 1.0*consistency_squared + 0.5*prompt_var.
            Defaults to "ccs_prompt_var".
        normalization: The kind of normalization to apply to the hidden states.
        num_layers: The number of layers in the MLP. Defaults to 1.
        pre_ln: Whether to include a LayerNorm module before the first linear
            layer. Defaults to False.
        supervised_weight: The weight of the supervised loss. Defaults to 0.0.

        lr: The learning rate to use. Ignored when `optimizer` is `"lbfgs"`.
            Defaults to 1e-2.
        num_epochs: The number of epochs to train for. Defaults to 1000.
        num_tries: The number of times to try training the reporter. Defaults to 10.
        optimizer: The optimizer to use. Defaults to "adam".
        weight_decay: The weight decay or L2 penalty to use. Defaults to 0.01.
    """

    activation: Literal["gelu", "relu", "swish"] = "gelu"
    bias: bool = True
    hidden_size: Optional[int] = None
    init: Literal["default", "pca", "spherical", "zero"] = "default"
    loss: list[str] = field(default_factory=lambda: ["ccs"])
    loss_dict: dict[str, float] = field(default_factory=dict, init=False)
    normalization: Literal["none", "meanonly", "full"] = "full"
    num_layers: int = 1
    pre_ln: bool = False
    seed: int = 42
    supervised_weight: float = 0.0

    lr: float = 1e-2
    num_epochs: int = 1000
    num_tries: int = 10
    optimizer: Literal["adam", "lbfgs"] = "lbfgs"
    weight_decay: float = 0.01

    def __post_init__(self):
        self.loss_dict = parse_loss(self.loss)

        # standardize the loss field
        self.loss = [f"{coef}*{name}" for name, coef in self.loss_dict.items()]


class CcsReporter(Reporter):
    """CCS reporter network.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    config: CcsReporterConfig

    def __init__(
        self,
        cfg: CcsReporterConfig,
        in_features: int,
        num_classes: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.config = cfg

        hidden_size = cfg.hidden_size or 4 * in_features // 3

        self.normalizers = [
            Normalizer(
                (in_features,), device=device, dtype=dtype, mode=cfg.normalization
            )
            for _ in range(num_classes)
        ]

        self.probe = nn.Sequential(
            nn.Linear(
                in_features,
                1 if cfg.num_layers < 2 else hidden_size,
                bias=cfg.bias,
                device=device,
            ),
        )
        if cfg.pre_ln:
            self.probe.insert(0, nn.LayerNorm(in_features, elementwise_affine=False))

        act_cls = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "swish": nn.SiLU,
        }[cfg.activation]

        for i in range(1, cfg.num_layers):
            self.probe.append(act_cls())
            self.probe.append(
                nn.Linear(
                    hidden_size,
                    1 if i == cfg.num_layers - 1 else hidden_size,
                    bias=cfg.bias,
                    device=device,
                )
            )

    def check_separability(
        self,
        train_hiddens: Hiddens,
        val_hiddens: Hiddens,
    ) -> float:
        """Measure how linearly separable the pseudo-labels are for a contrast pair.

        Args:
            train_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                unnormalized negative and positive representations respectively.
                Used for training the classifier.
            val_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                unnormalized negative and positive representations respectively.
                Used for evaluating the classifier.

        Returns:
            The AUROC of a linear classifier fit on the pseudo-labels.
        """

        train_hiddens_n = self.normalize(train_hiddens)
        val_hiddens_n = self.normalize(val_hiddens)

        model_dim = train_hiddens.shape[-1]
        pseudo_clf = Classifier(model_dim, device=train_hiddens.device)

        def get_pseudo_labels(x: list[Float[Tensor, "batch n_variants hidden_size"]]):
            n_batch = x[0].shape[0]
            n_variants = x[0].shape[1]

            pseudo_labels = torch.cat(
                [
                    x[0].new_ones(n_batch) * i
                    for i in range(len(x))
                ]
            )
            # make num_variants copies of each pseudo-label
            return pseudo_labels.repeat_interleave(n_variants)

        flatten_hiddens = lambda x: rearrange(
            x,
            "num_classes batch n_variants hidden_size -> (num_classes batch n_variants) hidden_size",
        )

        pseudo_train_labels = get_pseudo_labels(train_hiddens_n)
        pseudo_val_labels = get_pseudo_labels(val_hiddens_n)

        pseudo_clf.fit(
            flatten_hiddens(train_hiddens_n),
            pseudo_train_labels,
        )
        with torch.no_grad():
            pseudo_preds = pseudo_clf(flatten_hiddens(val_hiddens_n)).squeeze(-1)
            return roc_auc(pseudo_val_labels, pseudo_preds).item()

    def unsupervised_loss(self, logits: LogitsMultiChoice) -> Tensor:
        """Add together the losses specified in the `loss_dict`."""
        loss = sum(
            LOSSES[name](logits, coef) for name, coef in self.config.loss_dict.items()
        )
        return assert_type(Tensor, loss)

    def reset_parameters(self):
        """Reset the parameters of the probe.

        If init is "spherical", use the spherical initialization scheme.
        If init is "default", use the default PyTorch initialization scheme for
        nn.Linear (Kaiming uniform).
        If init is "zero", initialize all parameters to zero.
        """
        if self.config.init == "spherical":
            # Mathematically equivalent to the unusual initialization scheme used in
            # the original paper. They sample a Gaussian vector of dim in_features + 1,
            # normalize to the unit sphere, then add an extra all-ones dimension to the
            # input and compute the inner product. Here, we use nn.Linear with an
            # explicit bias term, but use the same initialization.
            assert len(self.probe) == 1, "Only linear probes can use spherical init"
            probe = cast(nn.Linear, self.probe[0])  # Pylance gets the type wrong here

            theta = torch.randn(1, probe.in_features + 1, device=probe.weight.device)
            theta /= theta.norm()
            probe.weight.data = theta[:, :-1]
            probe.bias.data = theta[:, -1]

        elif self.config.init == "default":
            for layer in self.probe:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

        elif self.config.init == "zero":
            for param in self.parameters():
                param.data.zero_()
        elif self.config.init != "pca":
            raise ValueError(f"Unknown init: {self.config.init}")

    def forward(self, x: Tensor) -> Tensor:
        """Return the raw score output of the probe on `x`."""
        assert x.shape[-2] == 2, "Probe input must be a contrast pair"

        # Apply normalization
        x0, x1 = x.unbind(-2)
        x0, x1 = self.neg_norm(x0), self.pos_norm(x1)
        x = torch.stack([x0, x1], dim=-2)

        return self.raw_forward(x)

    def raw_forward(self, x: Tensor) -> Tensor:
        """Apply the probe to the provided input, without normalization."""
        return self.probe(x).squeeze(-1)

    def loss(
        self,
        logits: LogitsMultiChoice,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the loss of the reporter on the contrast tuple (x0, ..., xk)
        which is typically a contrast pair (x0, x1) for binary tasks.

        Args:
            logits: the raw scores output by the reporter on the contrast pair.
            labels: The labels of the contrast pair. Defaults to None.

        Returns:
            loss: The loss of the reporter on the contrast tuple (x0, ..., xk).

        Raises:
            ValueError: If `supervised_weight > 0` but `labels` is None.
        """
        loss = self.unsupervised_loss(logits)

        # If labels are provided, use them to compute a supervised loss
        if labels is not None:
            num_labels = len(labels)
            assert num_labels <= len(logits), "Too many labels provided"
            num_variants = logits.shape[1]
            broadcast_labels = labels.repeat_interleave(num_variants)

            if logits.shape[-1] == 2:
                p0, p1 = logits[:num_labels].sigmoid().unbind(2)

                preds = p0.add(1 - p1).mul(0.5)
                # broadcast the labels, and flatten the predictions
                # so that both are 1D tensors
                flattened_preds = preds.cpu().flatten()
                supervised_loss = bce(
                    flattened_preds, broadcast_labels.type_as(flattened_preds)
                )

            else:
                p = logits[:num_labels].sigmoid()
                p_logits = p.log() - p.sum(dim=-1, keepdim=True).log()
                # manual cross-entropy calculation
                neg_p_logits = rearrange(
                    -p_logits, "batch n_variants k -> (batch n_variants) k"
                )
                supervised_loss = neg_p_logits[
                    torch.arange(len(neg_p_logits)), broadcast_labels.to(torch.long)
                ].mean()

            alpha = self.config.supervised_weight
            loss = alpha * supervised_loss + (1 - alpha) * loss

        elif self.config.supervised_weight > 0:
            raise ValueError(
                "Supervised weight > 0 but no labels provided to compute loss"
            )

        return loss

    def normalize(
        self, hiddens: Hiddens, fit=False
    ) -> list[Float[Tensor, "batch n_variants hidden_size"]]:
        classes = hiddens.unbind(2)
        assert len(classes) == len(self.normalizers), "Number of classes does not match"
        classes_n = []
        for normalizer, class_hiddens in zip(self.normalizers, classes):
            if fit:
                normalizer.fit(class_hiddens)
            classes_n.append(normalizer(class_hiddens))
        return classes_n


    def shuffle_labels(self, hiddens: Tensor, labels: Tensor) -> Tensor:
        """Shuffle the data according to the labels.
        That is, get the set of "true" data points and the set of "false" data points.
        Then create pairs of "true" and "false" data points.

        Args:
            hiddens: The hidden representations of the contrast pair.
            labels: The labels of the contrast pair.

        Returns:
            shuffled_hiddens: The shuffled hidden representations.
        """
        # Get the set of "true" data points and the set of "false" data points.
        true_hiddens = hiddens[labels == 1]
        false_hiddens = hiddens[labels == 0]
        if len(true_hiddens) != len(false_hiddens):
            warnings.warn(
                f"Number of true and false data points are not equal: {len(true_hiddens)} != {len(false_hiddens)}"
            )
            num_pairs = min(len(true_hiddens), len(false_hiddens))
            true_hiddens = true_hiddens[:num_pairs]
            false_hiddens = false_hiddens[:num_pairs]

        # Shuffle the "false" data points.
        shuffled_false_hiddens = false_hiddens[torch.randperm(len(false_hiddens))]

        # Create pairs of "true" and "false" data points.
        shuffled_hiddens = torch.cat([true_hiddens, shuffled_false_hiddens], dim=0)

        return shuffled_hiddens
        

    def fit(
        self,
        hiddens: Hiddens,
        labels: Optional[Tensor] = None,
    ) -> float:
        """Fit the probe to the contrast tuple, typically a pair (neg, pos).

        Args:
            hiddens: A batch of tensors, where the last dimension indexes the
                contrastive representations.
            labels: The labels of the contrast pair. Defaults to None.

        Returns:
            best_loss: The best loss obtained.

        Raises:
            ValueError: If `optimizer` is not "adam" or "lbfgs".
            RuntimeError: If the best loss is not finite.
        """
        classes_n = self.normalize(hiddens, fit=True)

        # Record the best acc, loss, and params found so far
        best_loss = torch.inf
        best_state: dict[str, Tensor] = {}  # State dict of the best run

        for i in range(self.config.num_tries):
            self.reset_parameters()

            # This is sort of inefficient but whatever
            if self.config.init == "pca":
                assert (
                    len(classes_n) == 2
                ), "PCA init can only be done for true/false tasks"
                x_neg, x_pos = classes_n
                diffs = rearrange(
                    x_pos - x_neg, "batch n_variants d -> (batch n_variants) d"
                )
                _, __, V = torch.pca_lowrank(diffs, q=i + 1)
                self.probe[0].weight.data = V[:, -1, None].T

            if self.config.optimizer == "lbfgs":
                loss = self.train_loop_lbfgs(hiddens, labels)
            elif self.config.optimizer == "adam":
                loss = self.train_loop_adam(hiddens, labels)
            else:
                raise ValueError(f"Optimizer {self.config.optimizer} is not supported")

            if loss < best_loss:
                best_loss = loss
                best_state = deepcopy(self.state_dict())

        if not math.isfinite(best_loss):
            raise RuntimeError("Got NaN/infinite loss during training")

        self.load_state_dict(best_state)
        return best_loss

    def train_loop_adam(
        self,
        hiddens: Hiddens,
        labels: Optional[Tensor] = None,
    ) -> float:
        """Adam train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        loss = torch.inf
        for _ in range(self.config.num_epochs):
            optimizer.zero_grad()

            loss = self.loss(self.raw_forward(hiddens), labels)
            loss.backward()
            optimizer.step()

        return float(loss)

    def train_loop_lbfgs(
        self,
        hiddens: Hiddens,
        labels: Optional[Tensor] = None,
    ) -> float:
        """LBFGS train loop, returning the final loss. Modifies params in-place."""

        eps = torch.finfo(hiddens.dtype).eps

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.config.num_epochs,
            tolerance_change=eps,
            tolerance_grad=eps,
        )
        # Raw unsupervised loss, WITHOUT regularization
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            loss = self.loss(self.raw_forward(hiddens), labels)
            regularizer = 0.0

            # We explicitly add L2 regularization to the loss, since LBFGS
            # doesn't have a weight_decay parameter
            for param in self.parameters():
                regularizer += self.config.weight_decay * param.norm() ** 2 / 2

            regularized = loss + regularizer
            regularized.backward()

            return float(regularized)

        optimizer.step(closure)
        return float(loss)
