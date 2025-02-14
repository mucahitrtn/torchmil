import torch
from torch import Tensor

from torchmil.nn.utils import masked_softmax, LazyLinear

class ProbSmoothAttentionPool(torch.nn.Module):
    def __init__(
        self,
        in_dim : int = None,
        att_dim : int = 128,
        covar_mode : str = 'diag',
        n_samples_train : int = 1000,
        n_samples_test : int = 5000,
    ) -> None:
        """
        Class constructor.

        Arguments:
            in_dim: Input dimension. If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            covar_mode: Covariance mode. Must be 'diag' or 'zero'.
            n_samples_train: Number of samples during training.
            n_samples_test: Number of samples during testing.        
        """
        super(ProbSmoothAttentionPool, self).__init__()
        self.covar_mode = covar_mode
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test

        if self.covar_mode not in ['diag', 'zero']:
            raise ValueError("covar_mode must be 'diag' or 'zero'")
                
        self.in_mlp = torch.nn.Sequential(
            LazyLinear(in_dim, 2*att_dim),
            torch.nn.GELU(),
            torch.nn.Linear(2*att_dim, 2*att_dim),
            torch.nn.GELU()
        )
        
        self.mu_f_nn = torch.nn.Linear(2*att_dim, 1)
    
        if self.covar_mode == 'diag':
            self.log_diag_Sigma_nn = torch.nn.Linear(2*att_dim, 1)

        self.eps = 1e-6
    
    def _sample_f(
        self, 
        mu_f : Tensor,
        log_diag_Sigma_f : Tensor,
        n_samples : int = 1
    ) -> Tensor:
        """
        Arguments:
            mu_f: Mean of q(f) of shape `(batch_size, bag_size, 1)`.
            log_diag_Sigma_f: Log diagonal of covariance of q(f) of shape `(batch_size, bag_size, 1)`.
            n_samples: Number of samples to draw.

        Returns:
            f: Sampled f of shape `(batch_size, bag_size, n_samples)`.
        """
        batch_size = mu_f.shape[0]

        if self.covar_mode == 'diag':
            bag_size = mu_f.shape[1]
            random_sample = torch.randn(batch_size, bag_size, n_samples, device=mu_f.device) # (batch_size, bag_size, n_samples)
            sqrt_diag_Sigma_f = torch.exp(0.5*log_diag_Sigma_f) # (batch_size, bag_size, 1)
            f = mu_f + sqrt_diag_Sigma_f*random_sample # (batch_size, bag_size, n_samples)
            f = torch.clip(f, -20, 20)
        else:
            f = mu_f
        return f

    def _kl_div(
        self, 
        mu_f : Tensor,
        log_diag_Sigma_f : Tensor,
        adj_mat : Tensor
    ) -> Tensor:
        """
        Arguments:
            mu_f: Mean of the attention distribution of shape `(batch_size, bag_size, 1)`.
            log_diag_Sigma_f: Log diagonal of covariance of the attention distribution of shape `(batch_size, bag_size, 1)`.
            adj_mat: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.

        Returns:
            kl_div: KL divergence of shape `()`.
        """

        bag_size = float(mu_f.shape[1])
        inv_bag_size = 1.0/bag_size

        if not adj_mat.is_sparse:
            adj_mat_dense = adj_mat
            diag_adj = torch.diagonal(adj_mat, dim1=1, dim2=2).unsqueeze(dim=-1) # (batch_size, bag_size, 1)
        else:
            adj_mat_dense = adj_mat.to('cpu').coalesce().to_dense()
            diag_adj = torch.diagonal(adj_mat_dense, dim1=1, dim2=2).unsqueeze(dim=-1).to(mu_f.device) # (batch_size, bag_size, 1)        

        muT_mu = torch.sum(mu_f**2, dim=(1,2)) # (batch_size,)
        adj_mat_mu = torch.bmm(adj_mat, mu_f) # (batch_size, bag_size, 1)
        muT_adjmat_mu = torch.bmm( mu_f.transpose(1,2), adj_mat_mu).squeeze(1,2) # (batch_size,)

        muT_lap_mu = inv_bag_size*(muT_mu - muT_adjmat_mu) # (batch_size,)

        if self.covar_mode == 'full':
            raise NotImplementedError("covar_mode='full' is not implemented yet")
        elif self.covar_mode == 'diag':
            diag_Sigma = torch.exp(log_diag_Sigma_f) # (batch_size, bag_size, 1)
            tr_Sigma = inv_bag_size*torch.sum(diag_Sigma, dim=(1,2)) # (batch_size,)
            tr_adj_Sigma = inv_bag_size*torch.sum(diag_adj*diag_Sigma, dim=(1,2)) # (batch_size,)
            log_det_Sigma = inv_bag_size*torch.sum(log_diag_Sigma_f, dim=(1,2)) # (batch_size,)
        else:
            tr_Sigma = torch.zeros((1,), device=mu_f.device)
            tr_adj_Sigma = torch.zeros((1,), device=mu_f.device)
            log_det_Sigma = torch.zeros((1,), device=mu_f.device)

        kl_div = torch.mean(muT_lap_mu + tr_Sigma - tr_adj_Sigma - 0.5*log_det_Sigma) # ()
        
        return kl_div
    
    def forward(
        self, 
        X : Tensor,
        adj : Tensor,
        mask : Tensor = None,
        return_att : bool = False,
        return_attdist : bool = False,
        return_kl_div : bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, D)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`. Only required when `return_kl_div=True`.
            return_att: If True, returns a sample from the attention distribution `f` in addition to `z`.
            return_attdist: If True, returns the attention distribution (`mu_f`, `diag_Sigma_f`) in addition to `z`.
            return_kl_div: If True, returns the KL divergence between the attention distribution and the prior distribution.
        
        Returns:
            z: Bag representation of shape `(batch_size, D)`.
            f: Sample from the attention distribution of shape `(batch_size, bag_size, n_samples)`. Only returned when `return_att=True`.
            mu_f: Mean of the attention distribution of shape `(batch_size, bag_size, 1)`. Only returned when `return_attdist=True`.
            diag_Sigma_f: Covariance of the attention distribution of shape `(batch_size, bag_size, 1)`. Only returned when `return_attdist=True`.
            kl_div: KL divergence between the attention distribution and the prior distribution, of shape `()`. Only returned when `return_kl_div=True`.
        """

        if self.training:
            n_samples = self.n_samples_train
        else:
            n_samples = self.n_samples_test
        
        batch_size = X.shape[0]
        bag_size = X.shape[1]
        
        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device) # (batch_size, bag_size)
        mask = mask.unsqueeze(dim=-1) # (batch_size, bag_size, 1)
        
        H = self.in_mlp(X) # (batch_size, bag_size, 2*att_dim)
        mu_f = self.mu_f_nn(H) # (batch_size, bag_size, 1)
        log_diag_Sigma_f = self.log_diag_Sigma_nn(H) # (batch_size, bag_size, 1)

        # sample from q(f)
        f = self._sample_f(mu_f, log_diag_Sigma_f, n_samples) # (batch_size, bag_size, n_samples)

        s = masked_softmax(f, mask) # (batch_size, bag_size, n_samples)

        z = torch.bmm(X.transpose(1,2), s) # (batch_size, d, n_samples)

        if return_kl_div:
            kl_div = self._kl_div(mu_f, log_diag_Sigma_f, adj) # ()
            if return_att:
                return z, f, kl_div
            elif return_attdist:
                return z, mu_f, log_diag_Sigma_f, kl_div
            else:
                return z, kl_div
        else:
            if return_att:
                return z, f
            elif return_attdist:
                return z, mu_f, log_diag_Sigma_f
            else:
                return z
