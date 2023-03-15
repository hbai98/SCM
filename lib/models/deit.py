from einops import rearrange, repeat
import torch
from torch.functional import einsum
import torch.nn as nn
from functools import partial

from .graphFusion import Fuse

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model
from timm.models import ResNet, Bottleneck
from timm.models.layers.classifier import create_classifier
import torch.nn.functional as F

__all__ = [
    'deit_scm_tiny_patch16_224', 'deit_scm_small_patch16_224', 'deit_scm_base_patch16_224',
    'deit_tscam_tiny_patch16_224', 'deit_tscam_small_patch16_224', 'deit_tscam_base_patch16_224',
    'vit_scm_small_patch16_224'
]


def embeddings_to_cosine_similarity_matrix(tokens):
    """
    Shapes for inputs:
    - tokens : :math:`(B, N, D)` where B is the batch size, N is the target `spatial` sequence length, D is the token representation length.

    Shapes for outputs:

    Converts a a tensor of D embeddings to an (N, N) tensor of similarities.
    """
    dot = torch.einsum('bij, bkj -> bik', [tokens, tokens])
    norm = torch.norm(tokens, p=2, dim=-1)
    x = torch.div(dot, torch.einsum('bi, bj -> bij', norm, norm))

    return x

class SCM(VisionTransformer):
    def __init__(self, num_layers=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.head = nn.Linear(self.embed_dim, self.num_classes)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes,
                              kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        encoder = Encoder(dim=self.num_classes)
        self.layers.append(encoder)
        # self.convs = ConvModules(num_layers+1, self.num_classes)
        for i in range(1, self.num_layers): 
            self.layers.append(Encoder(dim=self.num_classes, 
                                       fusion_cfg=dict(lapMat=encoder.fuse.laplacian, 
                                                 loss_rate=1, 
                                                 grid_size=(14, 14), 
                                                 iteration=4)))

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, test_select=0):
        x_cls, x_patch, attn = self.forward_features(x)
        n, p, c = x_patch.shape

        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)

        attn = torch.stack(attn)        # 12 * B * H * N * N
        attn = torch.mean(attn, dim=2)  # 12 * B * N * N
        
        n, c, h, w = x_patch.shape
        cams = attn.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
        cam = rearrange(cams, 'B 1 H W -> B (H W)')
        cam = norm_cam(cam)
        pred_cam = rearrange(cam, 'B (H W) -> B 1 H W', H=h)
        pred_semantic = x_patch
        
        if self.training:
            for i, layer in enumerate(self.layers):
                x_patch, cam = layer(x_patch, cam)
            x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)
            return x_logits
        else:
            x_logits = self.avgpool(pred_semantic).squeeze(3).squeeze(2)
            predict = pred_cam*pred_semantic
            if test_select>0:
                topk_ind = torch.topk(x_logits, test_select)[-1]
                predict = torch.tensor([torch.take(a, idx, axis=0) for (a, idx)  
                                        in zip(predict, topk_ind)])
                pred_semantic = torch.tensor([torch.take(a, idx, axis=0) for (a, idx)  
                                        in zip(pred_semantic, topk_ind)])
            return x_logits, predict, pred_cam, pred_semantic

class Encoder(nn.Module):
    def __init__(self,
                 dim,
                 thred=0.5,
                 residual=True,
                 fusion_cfg=dict(loss_rate=1, grid_size=(14, 14), iteration=16),
                 ) -> None:
        super().__init__()
        self.fuse = Fuse(**fusion_cfg)
        self.shrink = nn.Tanhshrink()
        self.thred = nn.Parameter(torch.ones([1])*thred)
        self.residual = residual
        H, W = fusion_cfg['grid_size']
        # self.norm = nn.LayerNorm((dim, H, W)) 

    def forward(self, x, cam):
        """foward function given x and spt

        Args:
            x (torch.Tensor): patch tokens, tensor of shape [B D H W]
            cam (torch.Tensor): cam values, tensor of shape [B N]
        Returns:
            x (torch.Tensor): patch tokens, tensor of shape [B D H W]
            cam (torch.Tensor): cam values, tensor of shape [B N]
        """
        sim = embeddings_to_cosine_similarity_matrix(
            rearrange(x, 'B D H W -> B (H W) D'))
        thred = self.thred.to(x.device)
        out_cam = einsum('b h w, b w -> b h', self.fuse(sim),cam)
        thred = thred * cam.max(1, keepdim=True)[0]
        out_cam = self.shrink(out_cam/thred)
        out_cam = norm_cam(out_cam)
        out_x = einsum('b d h w, b h w -> b d h w', x,
                       rearrange(out_cam, 'B (H W) -> B H W', H=x.shape[-2]))
        if self.residual:
            x = x + out_x
            cam = cam + out_cam
        return x, cam


def norm_cam(cam):
    # cam [B N]
    if len(cam.shape) == 3:
        cam = cam - repeat(rearrange(cam, 'B H W -> B (H W)').min(1,
                           keepdim=True)[0], 'B 1 -> B 1 1')
        cam = cam / repeat(rearrange(cam, 'B H W -> B (H W)').max(1,
                           keepdim=True)[0], 'B 1 -> B 1 1')
    elif len(cam.shape) == 2:
        cam = cam - cam.min(1, keepdim=True)[0]
        cam = cam / cam.max(1, keepdim=True)[0]
    elif len(cam.shape) == 4:
        # min-max norm for each class feature map
        B, C, H, W = cam.shape
        cam = rearrange(cam, 'B C H W -> (B C) (H W)')
        cam -= cam.min(1, keepdim=True)[0]
        cam /= cam.max(1, keepdim=True)[0]
        cam = rearrange(cam, '(B C) (H W) -> B C H W', B = B, H=H)
    return cam


class TSCAM(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes,
                              kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, return_cam=False):
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        if self.training:
            return x_logits
        else:
            attn_weights = torch.stack(
                attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            cams = attn_weights.sum(0)[:, 0, 1:].reshape(
                [n, h, w]).unsqueeze(1)
            cams = cams * feature_map                           # B * C * 14 * 14

            return x_logits, cams


@register_model
def deit_tscam_tiny_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k,
                           v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_tscam_small_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k,
                           v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_tscam_base_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k,
                           v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_scm_tiny_patch16_224(pretrained=False, **kwargs):
    model = SCM(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k,
                           v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_scm_small_patch16_224(pretrained=False, **kwargs):
    model = SCM(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k,
                           v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def vit_scm_small_patch16_224(pretrained=False, **kwargs):
    model = SCM(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., qkv_bias=True, qk_scale=768 ** -0.5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
        #     map_location="cpu", check_hash=True
        # )
        pre_vit_small = create_model('vit_small_patch16_224', pretrained=True)
        checkpoint = pre_vit_small.state_dict()

        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k,
                           v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def deit_scm_base_patch16_224(pretrained=False, **kwargs):
    model = SCM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k,
                           v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

