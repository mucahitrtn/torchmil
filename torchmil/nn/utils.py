import torch
import numpy as np

class LazyLinear(torch.nn.Module):
    """
    Lazy Linear layer. Extends `torch.nn.Linear` with lazy initialization.
    """
    def __init__(self, in_features=None, out_features=512, bias=True, device=None, dtype=None):
        super().__init__()

        if in_features is not None:
            self.module = torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        else:
            self.module = torch.nn.LazyLinear(out_features, bias=bias, device=device, dtype=dtype)
    
    def forward(self, x):
        return self.module(x)

def masked_softmax(
    X : torch.Tensor,
    mask : torch.Tensor = None,
    ) -> torch.Tensor:
    """
    Compute masked softmax along the second dimension.
    
    Arguments:
        X (Tensor): Input tensor of shape `(batch_size, N, ...)`.
        mask (Tensor): Mask of shape `(batch_size, N)`. If None, no masking is applied.
    
    Returns:
        Tensor: Masked softmax of shape `(batch_size, N, ...)`.
    """

    if mask is None:
        return torch.nn.functional.softmax(X, dim=1)

    while mask.dim() < X.dim():
        mask = mask.unsqueeze(-1)
        
    X_masked = X.masked_fill(mask == 0, -float('inf'))

    return torch.nn.functional.softmax(X_masked, dim=1)

class MaskedSoftmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, X, mask):
        """
        Compute masked softmax.
        
        Arguments:
            X (Tensor): Input tensor of shape `(batch_size, bag_size)`.
            mask (Tensor): Mask of shape `(batch_size, bag_size)`.
        
        Returns:
            Tensor: Masked softmax of shape `(batch_size, bag_size)`.
        """

        return masked_softmax(X, mask, dim=self.dim)
    
def get_feat_dim(
        feat_ext : torch.nn.Module,
        input_shape : tuple[int, ...]
    ) -> int:
    """
    Get feature dimension of a feature extractor.

    Arguments:
        feat_ext (torch.nn.Module): Feature extractor.
        input_shape (tuple): Input shape of the feature extractor.
    """
    with torch.no_grad():
        return feat_ext(torch.zeros((1, *input_shape))).shape[-1]


class SinusoidalPositionalEncodingND(torch.nn.Module):
    def __init__(self, n_dim, channels, dtype_override=None):
        """
        Positional encoding for tensors of arbitrary dimensions.

        Arguments:
            n_dim (int): Number of dimensions.
            channels (int): Number of channels.
            dtype_override (torch.dtype): Data type override.
        """
        super(SinusoidalPositionalEncodingND, self).__init__()
        self.n_dim = n_dim
        self.org_channels = channels
        channels = int(np.ceil(channels / (2*n_dim)) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.dtype_override = dtype_override
        self.channels = channels

        print("Channels: ", channels)
    
    def _get_embedding(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        Arguments:
            tensor (Tensor): Input tensor of shape `(batch_size, l1, l2, ..., lN, channels)`.
        
        Returns:
            Tensor: Positional encoding of shape `(batch_size, l1, l2, ..., lN, channels)`.
        """
        if len(tensor.shape) != self.n_dim + 2:
            raise RuntimeError("The input tensor has to be {}d!".format(self.n_dim + 2))

        shape = tensor.shape

        orig_ch = shape[-1]
        emb_shape = list(shape)[1:]
        emb_shape[-1] = self.channels * self.n_dim

        emb = torch.zeros(
            emb_shape,
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )

        for i in range(self.n_dim):
            pos = torch.arange(shape[i+1], device=tensor.device, dtype=self.inv_freq.dtype)
            sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
            emb_i = self._get_embedding(sin_inp)
            for _ in range(self.n_dim-i-1):
                emb_i = emb_i.unsqueeze(1)
            emb[..., i*self.channels : (i+1)*self.channels] = emb_i

        return emb[None, ..., :orig_ch].repeat(shape[0], *(1 for _ in range(self.n_dim)), 1)
    
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