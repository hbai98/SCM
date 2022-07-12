from typing import Iterable
from einops import rearrange


import torch
from torch import nn

from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import build_activation_layer

from mmcv.runner.base_module import BaseModule

import numpy as np

import copy
import itertools

from scipy.sparse.coo import coo_matrix
from scipy.sparse import diags


def inverse_schulz(X, iteration=5, alpha=0.002):
    """
    Computes an approximation of the matrix inversion using Newton-Schulz
    iterations
    Source NASA: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19920002505.pdf
    """
    assert len(X.shape) >= 2, "Can't compute inverse on non-matrix"
    assert X.shape[-1] == X.shape[-2], "Must be batches of square matrices"

    with torch.no_grad():
        device = X.device
        eye = torch.eye(X.shape[-1], device=device)
        # alpha should be sufficiently small to have convergence
        inverse = alpha * torch.transpose(X, dim0=-2, dim1=-1)

    for _ in range(iteration):
        inverse = inverse @ (2 * eye - X @ inverse)

    return inverse

class Fuse(nn.Module):
    """

    ----------
    Parameters:
    - grid_size : tuple of ints
    A patched feature map shape to build with.
    e.g. [W1, ..., WN] where ：
    Wi - patch number of the axis i

    - num_connect : int
    the number of neighbor units to fuse against.

    - dilation : int
    the step size for fusion.

    - adjMat: coo_matrix
    sparse coordinate matrix for adjacent positions relationship.
    Matrix is available automatically after the model initialization.
    Save and assign a matrix if the attention shape not change for reducing space cost.

    - idMat: coo_matrix
    sparse coordinate matrix for in-degree relationship.
    Matrix is available automatically after the model initialization.
    Save and assign a matrix if the attention shape not change for reducing space cost.

    - init_cfg (dict, optional): Initialization config dict. Default to None
    """

    def __init__(self,
                 grid_size: Iterable[int],
                 iteration: int,
                 num_connect: int = 4,
                 dilation: int = 1,
                 adjMat: coo_matrix = None,
                 idMat: coo_matrix = None,
                 lapMat: coo_matrix = None,
                 loss_rate: float = 1, 
                 init_cfg: dict = None, ):
        super(Fuse, self).__init__()

        self._grid_size = grid_size
        self.num_patch = np.prod(self.grid_size)
        self.dimension = len(grid_size)
        self.iteration = iteration

        self._loss_rate = nn.Parameter(torch.ones([1])*loss_rate)

        if num_connect is None:
            self._num_connect = self.dimension * 2
        elif num_connect < 0:
            raise ValueError(
                f'Expect connections per unit is positive, but got {num_connect} instead.'
            )
        else:
            self._num_connect = num_connect

        self._dilation = dilation
        self._adjMat = adjMat
        self._idMat = idMat
        self._lap = lapMat
        
        self._lap = self.getLap()
        self.init_weights()

    def getAdj(self) -> coo_matrix:
        if self._adjMat is not None:
            return self._adjMat
        # patch idx array
        idx_row = []
        idx_col = []
 
        for idx_r in range(self.num_patch):
            idxes = self.getDimIdx(idx_r)
            id_c = [[idx - self._dilation, idx + self._dilation]
                    for idx in idxes]
            id_c = self.selectValidIdx(self.connect(idxes, id_c))
            idp_c = [self.getPatchIdx(i) for i in id_c]

            for idx_c in idp_c:
                idx_row.append(idx_r)
                idx_col.append(idx_c)

        self._adjMat = coo_matrix((np.full(len(idx_row), 1, dtype=int), (idx_row, idx_col)), shape=(
            self.num_patch, self.num_patch), dtype=int)

        return self._adjMat

    def getIdMat(self):
        '''
        Get in-degree matrix. Row is the in-direction.
        '''
        if self._idMat is not None:
            return self._idMat
        self._adjMat = self.getAdj()
        self._idMat = diags(self._adjMat.sum(axis=0).A1)

        return self._idMat

    def getLap(self):
        if self.laplacian is not None:
            return self.laplacian
        if self.adjMat is None:
            self.adjMat = self.getAdj()
        if self.idMat is None:
            self.idMat = self.getIdMat()

        lap = self.idMat - self.adjMat
        return lap

    def getPatchIdx(self, idxes):
        '''
        Get patch index by dimension indexes, e.g. idx_patch_D_W_H (2, 10, 9) = 2*s1*s2 + 10*s2 + 9
        where s0, s1, s2 is the maximum number of patches along axies.

        Without access and output safety check
        '''
        idx_axis = [idxes[i]*np.prod(self._grid_size[1+i:])
                    for i in range(self.dimension)]
        return np.sum(idx_axis, dtype=int)

    def getDimIdx(self, idx):
        '''
        Get dimension index by patch idx, e.g. idx_dim_D_W_H (10) = 10/(s1*s2), left/s2, left'
        where s0, s1, s2 is the maximum number of patches along axies.

        Without access and output safety check
        '''
        left = idx
        idxes = []
        for i in range(self.dimension):
            prod_ = np.prod(self._grid_size[1+i:], dtype=int)
            idx_ = left//prod_
            left -= idx_*prod_
            idxes.append(idx_)
        return idxes

    def connect(self, idxes, idxes_ref):
        '''
        Connect neighbors based on num_connect within dilation range.

        Parameters
        -------
        idxes: the original source indexes. e.g. [idx0, ..., idxn-1]
        idxes_ref: indexes after dilation. e.g. [(idx0 - self.dilation, idx0 + self.dilation) ...] from dim 0 to dim n-1.

        Returns
        -------
        A list of coordinates of range to fuse against.

        '''
        res = []

        # direction 1: extend range along each axis
        if self._num_connect == self.dimension*2:
            for i, ref in enumerate(idxes_ref):

                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]

                res.append(idxes_1)
                res.append(idxes_2)

        # direction 2: combine all conners
        elif self._num_connect == self.dimension*2 + np.power(2, self.dimension):
            idxes = list(itertools.product(*idxes_ref))
            res = idxes

        # direction 3: full-range np.power([1+self.dilation*2, 2*self.dilation-1], np.repeat(self.dimension, 2))
        elif self._num_connect == np.power(1+self._dilation*2, self.dimension) - np.power(self._dilation*2-1, self.dimension):
            res = set()
            for i, ref in enumerate(idxes_ref):
                idxes_ = copy.deepcopy(idxes_ref)
                del idxes_[i]
                pf = itertools.product(
                    *[list(range(r[0], r[1]+1)) for r in idxes_])
                idxes = []
                for pf_ in pf:
                    pf_1 = list(pf_)
                    pf_2 = list(pf_)
                    pf_1.insert(i, ref[0])
                    pf_2.insert(i, ref[1])
                    idxes.append(pf_1)
                    idxes.append(pf_2)
                res.update(idxes)
            res = list(idxes)

        else:
            raise ValueError(
                f'num_connect {self._num_connect} is not defined in your grid shape settings {self.grid_size}, which is a {len(self.grid_size)}D space.')
        return res

    def selectValidIdx(self, idxes):
        res = list(zip(*idxes))
        sel = np.full(len(idxes), True, dtype=bool)
        for i, idx in enumerate(res):
            a = np.array(idx)
            sel = sel & (a >= 0) & (a < self._grid_size[i])

        return np.array(idxes)[sel]

    @property
    def num_connect(self):
        return self._num_connect

    @num_connect.setter
    def num_connect(self, val):
        self._num_connect = val

    @property
    def dilation(self):
        return self._dilation

    @dilation.setter
    def dilation(self, val):
        self._dilation = val

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def adjMat(self):
        return self._adjMat

    @adjMat.setter
    def adjMat(self, val):
        self._adjMat = val

    @property
    def idMat(self):
        return self._idMat

    @idMat.setter
    def idMat(self, val):
        self._idMat = val

    @property
    def laplacian(self):
        return self._lap

    @laplacian.setter
    def laplacian(self, val):
        self._lap = val

    @property
    def loss_rate(self):
        return self._loss_rate

    @loss_rate.setter
    def loss_rate(self, val):
        self._loss_rate = nn.Parameter(torch.empty([1])*val)

    def forward(self, sim):
        r"""Allows the model to generate the fusion-based attention matrix.
        Fuse one time -> one iteration only.
    Args:
        sim: patch pair wise similarity matrix.

        Shapes for inputs:
        - sim: :math:`(B, N, N)`, where B is the batch size, N is the target `spatial` sequence length.
        Shapes for outputs:
        - fAttn_output: :math:`(B, N, N)` where B is the batch size, N is the target `spatial` sequence length.

    Examples:

        >>> d
    """
        if len(sim.shape) != 3:
            raise ValueError(
                f'Expect the patch pair-wise similarity matrix\'s shape to be [B, N, N], but got {sim.shape} instead.'
            )
        assert sim.shape[-1] == self.num_patch and sim.shape[-1] == sim.shape[-2], f'Expect he patch pair-wise similarity matrix to have {self.num_patch} tokens, but got {sim.shape[-1]}.'

        # TODO: test the module functionality
        with torch.no_grad():
            factory_kwargs = {'device': sim.device, 'dtype': sim.dtype}
            L = torch.tensor(self.laplacian.todense().A, **factory_kwargs)
            # lr = torch.sigmoid(self.loss_rate.to(factory_kwargs['device']))
            lr = self.loss_rate.to(factory_kwargs['device'])
            
        L = torch.mul(L, lr*sim - 1)
        L = inverse_schulz(L, iteration=self.iteration)
        L = L.transpose(dim0=-2, dim1=-1)

        return L

    def init_weights(self):

        pass
        # nn.init.constant_(self.loss_rate, 0)

    def __repr__(self):
        s = super().__repr__()
        s = s[:-2]
        s += '\n  fusion_cfg:('
        s += f'\n    grid_size={self.grid_size}'
        s += f'\n    dilation={self.dilation}'
        s += f'\n    num_connect={self.num_connect}'
        s += f'\n    loss_rate={self.loss_rate}'
        s += '\n))'
        return s

class fMultiheadAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None,
                 fusion_cfg=dict(grid_size=(14, 14))):
        super(fMultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut
        self.num_patch = np.prod(fusion_cfg['grid_size'])

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.pf = nn.Parameter(torch.ones([1]))
        self.fuse = Fuse(**fusion_cfg)

        self.v_pj = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.attn = nn.Linear(self.input_dims, self.num_patch*self.num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_act = build_activation_layer(dict(type='GELU'))
        
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = build_dropout(dropout_layer)
        
        # self.layerscale_gamma = nn.Parameter()

    def forward(self, x):
        B, N, _ = x.shape
        v = rearrange(self.v_pj(x), 'B N (h D) -> B h N D', h=self.num_heads)
        attn = rearrange(self.attn(x), 'B N (n h) -> B h N n', h=self.num_heads, n=self.num_patch)
        attn = self.attn_act(attn)
        attn = attn.softmax(dim=-1)
        
        fAttn = self.fuse(rearrange(attn, 'B h W H -> (B h) W H'))
        fAttn = rearrange(fAttn, '(B h) H W -> B h H W', h=self.num_heads)
        fAttn = fAttn*torch.sigmoid(self.pf) + attn*(1-torch.sigmoid(self.pf))
        fAttn = fAttn.softmax(dim=-1)
        fAttn = self.attn_drop(fAttn)       
        
        fx = (fAttn @ v).transpose(1, 2).reshape(B, N, self.embed_dims) 
        
        x = self.proj(fx)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

