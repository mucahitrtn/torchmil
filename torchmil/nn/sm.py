from typing import Union

import torch

class Sm(torch.nn.Module):
    r"""
    The $\texttt{Sm}$ operator, proposed in the paper [$\texttt{Sm}$: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).

    Given an input graph with node features $\mathbf{U} \in \mathbb{R}^{N \times D}$ and adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, in the exact mode the $\texttt{Sm}$ operator is defined as:

    \begin{align}
        \texttt{Sm}(\mathbf{U}) = ( \mathbf{I} + \gamma \mathbf{L} )^{-1} \mathbf{U},
    \end{align}

    where $\gamma \in (0, \infty)$ is a hyperparameter, $\mathbf{L} = \mathbf{D} - \mathbf{A}$ is the graph Laplacian, and $\mathbf{D}$ is the degree matrix.
    If `mode='approx'`, the $\texttt{Sm}$ operator is approximated as $\texttt{Sm}(\mathbf{U}) = G(T)$, where

    \begin{align}
        G(0) = \mathbf{U}, \quad G(t) = \alpha ( \mathbf{I} - \mathbf{L} ) G(t-1) + (1-\alpha) \mathbf{U},
    \end{align}

    for $t \in \{1, \ldots, T\}$, and $\alpha \in (0, 1)$ is a hyperparameter.

    """
    def __init__(
        self,
        alpha : Union[float, str] = 'trainable',
        num_steps : int = 10,
        mode : str = 'approx'
    ) -> None:
        """
        Arguments:
            alpha: Alpha value for the Sm operator. If 'trainable', alpha is a trainable parameter.
            num_steps: Number of steps to approximate the exact Sm operator.
            mode: Mode of the Sm operator. Possible values: 'approx', 'exact'.
        """
        super().__init__()
        self.alpha = alpha
        self.num_steps = num_steps
        self.mode = mode

        if self.mode == 'approx':
            self.sm = ApproxSm(alpha=alpha, num_steps=num_steps)
        elif self.mode == 'exact':
            self.sm = ExactSm(alpha=alpha)
        else:
            raise ValueError("mode must be 'approx' or 'exact'")

    def forward(
        self,
        f : torch.Tensor,
        adj_mat : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            f: Input tensor of shape `(batch_size, bag_size, ...)`.
            adj_mat: Adjacency matrix tensor of shape `(batch_size, bag_size, bag_size)`. Sparse tensor is supported.

        Returns:
            g: Output tensor of shape `(batch_size, bag_size, ...)`.
        """
        g = self.sm(f, adj_mat)
        return g

class ApproxSm(torch.nn.Module):
    r"""
    $\texttt{Sm}$ operator in the approximate mode, proposed in the paper [$\texttt{Sm}$: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).

    Given an input graph with node features $\mathbf{U} \in \mathbb{R}^{N \times D}$ and adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, it computes $\texttt{Sm}(\mathbf{U}) = G(T)$, where

    \begin{align}
        G(0) = \mathbf{U}, \quad G(t) = \alpha ( \mathbf{I} - \mathbf{L} ) G(t-1) + (1-\alpha) \mathbf{U},
    \end{align}

    for $t \in \{1, \ldots, T\}$, and $\alpha \in (0, 1)$ is a hyperparameter.

    """
    def __init__(
        self,
        alpha : Union[float, str] = 'trainable',
        num_steps : int = 10
    ) -> None:
        """
        Arguments:
            alpha: Alpha value for the Sm operator. If 'trainable', alpha is a trainable parameter.
            num_steps: Number of steps to approximate the exact Sm operator.
        """
        super().__init__()
        self.alpha = alpha
        self.num_steps = num_steps

        if isinstance(self.alpha, float):
            self.coef = (1.0/(1.0-self.alpha)-1)
        elif self.alpha == 'trainable':
            self.coef = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            raise ValueError("alpha must be float or 'trainable'")

    def forward(
        self,
        f : torch.Tensor,
        adj_mat : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            f: Input tensor of shape `(batch_size, bag_size, ...)`.
            adj_mat: Adjacency matrix tensor of shape `(batch_size, bag_size, bag_size)`. Sparse tensor is supported.

        Returns:
            g: Output tensor of shape `(batch_size, bag_size, ...)`.
        """

        # torch.sparse bug: this is a workaround
        recover_f = False
        if f.shape[2] == 1:
            recover_f = True
            f = torch.stack([f, f], dim=2).squeeze(-1) # (batch_size, bag_size, 2)

        g = f
        alpha = 1.0 / (1.0 + self.coef)
        for _ in range(self.num_steps):
            g = (1.0 - alpha)*f + alpha*torch.bmm(adj_mat, g) # (batch_size, bag_size, ...)

        if recover_f:
            g = g[:, :, 0].unsqueeze(-1) # (batch_size, bag_size, 1)

        return g

class ExactSm(torch.nn.Module):

    r"""
    $\texttt{Sm}$ operator in the exact mode, proposed in the paper [$\texttt{Sm}$: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).

    Given an input graph with node features $\mathbf{U} \in \mathbb{R}^{N \times D}$ and adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, it computes

    \begin{align}
        \texttt{Sm}(\mathbf{U}) = ( \mathbf{I} + \gamma \mathbf{L} )^{-1} \mathbf{U},
    \end{align}

    where $\gamma \in (0, \infty)$ is a hyperparameter, $\mathbf{L} = \mathbf{D} - \mathbf{A}$ is the graph Laplacian, and $\mathbf{D}$ is the degree matrix.


    """

    def __init__(
        self,
        alpha : Union[float, str] = 'trainable'
    ) -> None:
        """
        Arguments:
            alpha: Alpha value for the Sm operator. If 'trainable', alpha is a trainable parameter.
        """
        super().__init__()
        self.alpha = alpha

        if isinstance(self.alpha, float):
            self.coef = (1.0/(1.0-self.alpha)-1)
        elif self.alpha == 'trainable':
            self.coef = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            raise ValueError("alpha must be float or 'trainable'")

    def forward(
        self,
        f : torch.Tensor,
        adj_mat : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            f: Input tensor of shape `(batch_size, bag_size, ...)`.
            adj_mat: Adjacency matrix tensor of shape `(batch_size, bag_size, bag_size)`.

        Returns:
            g: Output tensor of shape `(batch_size, bag_size, ...)`.
        """
        batch_size = f.shape[0]
        bag_size = f.shape[1]

        id_mat = torch.eye(bag_size, device=adj_mat.device).unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, bag_size, bag_size)

        M = (1+self.coef)*id_mat - self.coef*adj_mat # (batch_size, bag_size, bag_size)
        g = self._solve_system(M, f) # (batch_size, bag_size, d_dim)
        return g

    def _solve_system(
        self,
        A : torch.Tensor,
        b : torch.Tensor
        ):
        """
        Solve the system Ax = b.

        Arguments:
            A: Matrix of shape `(batch_size, n, n)`.
            b: Vector of shape `(batch_size, n, ...)`.

        Returns:
            x: Solution of the system of shape `(batch_size, n, ...)`.
        """
        x = torch.linalg.solve(A, b)
        return x
