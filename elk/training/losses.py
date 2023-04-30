"""Loss functions for training reporters."""

import warnings
from inspect import signature, get_annotations

import torch
from torch import Tensor
from jaxtyping import Float

LOSSES = dict()  # Registry of loss functions


LogitsTrueFalse = Float[Tensor, "*batch n_variants 2"]

LogitsMultiChoice = Float[Tensor, "*batch n_variants k"]

def check_true_false(logits: Tensor, message = "This loss only works for true/false answers."):
    assert logits.shape[-1] == 2, message


def register(name):
    """A decorator to register a function to LOSSES"""

    def decorate(func):
        assert signature(func).parameters.keys() == {"logits", "coef"}, (
            f"Loss function {func.__name__} must take arguments "
            "`logits` and `coef`."
        )
        assert (
            name not in LOSSES
        ), f"Loss function {name} conflicts with existing function."
        LOSSES[name] = func
        return func

    return decorate


def H(p: Tensor) -> Tensor:
    """Entropy of Bernoulli distribution(s) with success probability `p`."""
    return torch.nn.functional.binary_cross_entropy(p, p)


@register("ccs")
def ccs_squared_loss(logits: LogitsTrueFalse, coef: float = 1.0) -> Tensor:
    """CCS loss from original paper, with squared differences between probabilities.

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logits: The log odds for the true and false answer completions.
        coef: The coefficient to multiply the loss by.
    Returns:
        The sum of the consistency and confidence losses.
    """
    check_true_false(logits)
    loss = consistency_squared_loss(logits) + confidence_squared_loss(logits)
    return coef * loss


@register("multi_ccs")
def multi_ccs_squared_loss(logits: LogitsMultiChoice, coef: float = 1.0) -> Tensor:
    """A generalization of CCS loss from the original paper, modified for multiple choices.

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logits: The stacked log odds for each of the log odds.
        coef: The coefficient to multiply the loss by.
    Returns:
        The sum of the consistency and confidence losses.
    """
    loss = multi_consistency_squared_loss(logits) + multi_confidence_squared_loss(logits)
    return coef * loss


@register("multi_consistency_squared")
def multi_consistency_squared_loss(
    logits: LogitsMultiChoice,
    coef: float = 1.0,
) -> Tensor:
    """Negation consistency loss based on the squared difference between the
    two distributions."""
    p = logits.sigmoid()
    return coef * (1 - p.sum(dim=-1)).square().mean()


@register("multi_confidence_squared")
def multi_confidence_squared_loss(
    logits: LogitsMultiChoice,
    coef: float = 1.0,
) -> Tensor:
    """Confidence loss based on the squared difference between the two distributions."""
    p = logits.sigmoid()
    return coef * (1 - p.max(dim=-1)).square().mean()


@register("multi_se_loss")
def multi_squared_error_loss(logits: LogitsMultiChoice, coef: float = 1.0) -> Tensor:
    """Tries to get the lower (k-1) logits to be close to 0, and the largest to be close to 1, via squared error.

    Args:
        logits: The stacked log odds for each of the log odds.
        coef: The coefficient to multiply the loss by.
    Returns:
        loss.
    """
    max_p = logits.max(dim=-1)
    loss = (
        logits.square().sum(dim=-1)
        - max_p.square()
        + (1 - max_p).square()
    ).mean()
    return coef * loss


@register("multi_log_loss")
def multi_log_error_loss(logits: LogitsMultiChoice, coef: float = 1.0) -> Tensor:
    """Tries to get the lower (k-1) logits to be close to 0, and the largest to be close to 1, via log(1-x).

    Args:
        logits: The stacked log odds for each of the log odds.
        coef: The coefficient to multiply the loss by.
    Returns:
        loss.
    """
    loss = -(
        (1 - logits).log().sum(dim=0)
        - (1 - logits.max(dim=0)).log()
        + logits.max(dim=0).log()
    ).mean()
    return coef * loss


@register("ccs_prompt_var")
def ccs_prompt_var_loss(logits: LogitsTrueFalse, coef: float = 1.0) -> Tensor:
    """CCS loss with prompt variance regularization.

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logit0: The log odds for the original proposition. Shape ([batch,] n_variants)
        logit1: The log odds for the negated proposition. Shape ([batch,] n_variants)
        coef: The coefficient to multiply the loss by.
    Returns:
        The sum of the consistency and confidence losses.
    """
    check_true_false(logits)
    loss = (
        consistency_squared_loss(logits)
        + confidence_squared_loss(logits)
        + prompt_var_loss(logits)
    )
    return coef * loss


@register("js")
def js_loss(
    logits: LogitsTrueFalse,
    coef: float = 1.0,
) -> Tensor:
    """Negation consistency loss based on the Jensen-Shannon divergence.

    Note that by default we use the base 2 logarithm, so the value is measured in bits.
    This ensures the divergence is in the range [0, 1]."""
    check_true_false(logits)
    p0, p1 = logits.sigmoid().unbind(dim=-1)
    neg_p1 = 1 - p1
    nats = H((p0 + neg_p1) / 2) - (H(p0) + H(neg_p1)) / 2
    return coef * nats


@register("js_confidence")
def js_confidence_loss(
    logits: LogitsTrueFalse,
    coef: float = 1.0,
) -> Tensor:
    """Confidence loss based on the Jensen-Shannon divergence. This is the same as the
    entropy of the 50/50 mixture of the two distributions.

    Note that by default we use the base 2 logarithm, so the value is measured in bits.
    This ensures the divergence is in the range [0, 1]."""
    check_true_false(logits)
    p0, p1 = logits.sigmoid().unbind(dim=-1)
    neg_p1 = 1 - p1
    nats = H((p0 + neg_p1) / 2)
    return coef * nats


@register("consistency_squared")
def consistency_squared_loss(
    logits: LogitsTrueFalse,
    coef: float = 1.0,
) -> Tensor:
    """Negation consistency loss based on the squared difference between the
    two distributions."""
    check_true_false(logits)
    p0, p1 = logits.sigmoid().unbind(-1)
    return coef * p0.sub(1 - p1).square().mean()


@register("confidence_squared")
def confidence_squared_loss(
    logits: LogitsTrueFalse,
    coef: float = 1.0,
) -> Tensor:
    """Confidence loss based on the squared difference between the two distributions."""
    check_true_false(logits)
    p0, p1 = logits.sigmoid().unbind(-1)
    return coef * torch.min(p0, p1).square().mean()


@register("prompt_var_squared")
def prompt_var_loss(logits: LogitsTrueFalse, coef: float = 1.0) -> Tensor:
    """
    Prompt-variance loss: the squared difference between the probability
    of a proposition and the mean probability over all variants of that
    proposition (templates).

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logit0: The log odds for the original proposition. shape ([batch,] n_variants)
        logit1: The log odds for the negated proposition. shape ([batch,] n_variants)
        coef: The coefficient to multiply the loss by.
    """
    if logits.shape[-2] == 1:
        warnings.warn(
            "Only one variant provided. Prompt variance loss will cause errors."
        )

    p0, p1 = logits.sigmoid().unbind(-1)

    var0 = p0.var(dim=-1, unbiased=False).mean()
    var1 = p1.var(dim=-1, unbiased=False).mean()
    prompt_variance = var0 + var1
    return coef * prompt_variance
