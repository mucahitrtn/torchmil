import torch
import numpy as np

def log_sum_exp(x):
    """
    Compute log(sum(exp(x), 1)) in a numerically stable way.
    Assumes x is 2d.
    """
    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))

def delta(y, labels, alpha=None):
    """
    Compute zero-one loss matrix for a vector of ground truth y
    """

    if isinstance(y, torch.autograd.Variable):
        labels = torch.autograd.Variable(labels, requires_grad=False).to(y.device)

    delta = torch.ne(y[:, None], labels[None, :]).float()

    if alpha is not None:
        delta = alpha * delta
    return delta

def Top1_Hard_SVM(labels, alpha=1.):
    def fun(x, y):
        y = y.long()
        # max oracle
        max_, _ = (x + delta(y, labels, alpha)).max(1)
        # subtract ground truth
        loss = max_ - x.gather(1, y).squeeze()
        return loss
    return fun

def Top1_Smooth_SVM(labels, tau, alpha=1.):
    def fun(x, y):
        # add loss term and subtract ground truth score
        y = y.long()
        x = x + delta(y, labels, alpha) - x.gather(1, y)
        # compute loss
        loss = tau * log_sum_exp(x / tau)

        return loss
    return fun

def detect_large(x, k, tau, thresh):
    top, _ = x.topk(k + 1, 1)
    # switch to hard top-k if (k+1)-largest element is much smaller than k-largest element
    hard = torch.ge(top[:, k - 1] - top[:, k], k * tau * np.log(thresh)).detach()
    smooth = hard.eq(0)
    return smooth, hard

class _SVMLoss(torch.nn.Module):

    def __init__(
        self, 
        n_classes : int,
        alpha : float = 1.0
    ) -> None:
        """

        Arguments:
            n_classes: Number of classes.
            alpha: Regularization parameter.       
        """

        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.n_classes = n_classes
        self._tau = None

    def forward(self, x, y):
        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self._tau != tau:
            # print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        torch.nn.Module.cuda(self, device)
        self.get_losses()
        return self

    def cpu(self):
        torch.nn.Module.cpu(self)
        self.get_losses()
        return self

    def get_losses(self):
        raise NotImplementedError("get_losses needs to be re-implemented for each loss")

class SmoothTop1SVM(_SVMLoss):
    def __init__(
        self, 
        n_classes : int,
        alpha : float = 1.0,
        tau : float = 1.0
    ) -> None:
        """
        Smooth Top-1 SVM loss, as described in [Smooth Loss Functions for Deep Top-k Classification](https://arxiv.org/abs/1802.07595).
        Implementation adapted from [the original code](https://github.com/oval-group/smooth-topk).

        Arguments:
            n_classes: Number of classes.
            alpha: Regularization parameter.
            tau: Temperature parameter.
        """
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes, alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(
        self, 
        x : torch.Tensor,
        y : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, n_classes)`. If `n_classes=1`, the tensor is assumed to be the positive class score.
            y: Target tensor of shape `(batch_size,)`.
        
        Returns:
            loss: Loss tensor of shape `(batch_size,)`.
        """
        
        if x.shape[1] == 1:
            x = torch.cat([x, -x], 1) # add dummy dimension for binary classification

        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)

        return loss

    def get_losses(self):
        labels = torch.from_numpy(np.arange(self.n_classes))
        self.F_h = Top1_Hard_SVM(labels, self.alpha)
        self.F_s = Top1_Smooth_SVM(labels, self.tau, self.alpha)