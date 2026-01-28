import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules import Transformer
from typing import Optional, Any, Union, Callable, Tuple
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.parameter import Parameter
from torch.nn.modules import LayerNorm, Linear, Dropout
from torch.nn.modules.container import ModuleList
# from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import copy

class Transformer(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu) -> None:
        super().__init__()
        
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        
    def forward(self, src, tgt) -> Tensor:
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        
        return output
    
class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        
        for mod in self.layers:
            output = mod(output)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output    

class TransformerEncoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation
    
    def forward(self, src: Tensor) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))
        
        return x
    
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x)[0]
        
        return self.dropout1(x)
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        output = tgt
        
        for mod in self.layers:
            output = mod(output, memory)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output

class TransformerDecoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        super().__init__()  
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = torch.nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.activation = activation
        
    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        x = tgt
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._mha_block(x, memory))
        x = self.norm3(x + self._ff_block(x))
        
        return x
    
    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)
    
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor) -> Tensor:
        x = self.multihead_attn(x, mem, mem)[0]
        return self.dropout2(x)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
        
class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # W_q, W_k, W_v
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None


        self._reset_parameters()
    
    # Initiate the Member Variable
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_uniform_(self.bias_k)
        if self.bias_v is not None:
            xavier_uniform_(self.bias_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = False, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True, is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                # key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                # attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                # average_attn_weights=average_attn_weights,
                # is_causal=is_causal
            )    
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                # key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                # attn_mask=attn_mask,
                # average_attn_weights=average_attn_weights,
                # is_causal=is_causal
            )    

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
        
def multi_head_attention_forward(
    query: Tensor, key: Tensor, value: Tensor, 
    embed_dim_to_check: int, num_heads: int,
    in_proj_weight: Optional[Tensor], in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor], bias_v: Optional[Tensor], dropout_p: float,
    out_proj_weight: Tensor, out_proj_bias: Optional[Tensor], training: bool = True,
    need_weights: bool = False, use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    
    assert embed_dim == embed_dim_to_check
    if isinstance(embed_dim, torch.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim
    
    # projection
    if not use_separate_proj_weight:
        assert in_proj_bias is not None
        q, k, v = in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None
        assert k_proj_weight is not None
        assert v_proj_weight is not None
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q = b_k = b_v = in_proj_bias.chunk(3)
        q, k, v = in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])

    # reshape q, k, v
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if not training:
        dropout_p = 0.0
    
    if need_weights:
        pass
    else:
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, None, dropout_p, False)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return attn_output, None



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def in_projection_packed(q: Tensor, k: Tensor, v: Tensor, w: Tensor, b: Optional[Tensor] = None) -> list[Tensor]:
    E = q.size(-1)
    # q = k = v
    proj = F.linear(q, w, b)
    proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    return proj[0], proj[1], proj[2]

def in_projection(q: Tensor, k: Tensor, v: Tensor,
                   w_q: Tensor, w_k: Tensor, w_v: Tensor,
                   b_q: Optional[Tensor] = None,
                   b_k: Optional[Tensor] = None,
                   b_v: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq)
    assert w_k.shape == (Eq, Ek)
    assert w_v.shape == (Eq, Ev)
    assert b_q is None or b_q.shape == (Eq, )
    assert b_k is None or b_k.shape == (Eq, )
    assert b_v is None or b_v.shape == (Eq, )
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def test() -> None:
    my_transformer = Transformer(d_model=512, nhead=8)
    official_transformer = torch.nn.modules.Transformer(d_model=512, nhead=8)

    try:
        my_transformer.load_state_dict(official_transformer.state_dict())
    except RuntimeError as e:
        print(e)

    my_transformer.eval()
    official_transformer.eval()


    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    with torch.no_grad():
        official_output = official_transformer(src, tgt)
        my_output = my_transformer(src, tgt)

    is_close = torch.allclose(official_output, my_output, atol=1e-5)
    if is_close:
        print("🎉 测试通过！你的实现与官方完全一致。")  
    else:
        print("⚠️ 测试未通过。输出数值存在差异。")

def export() -> None:
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    model = Transformer(d_model=512, nhead=8)
    
    model.eval()
    
    torch.onnx.export(
            model, 
            (src, tgt),                  # 输入的 Tuple
            "transformer.onnx",          # 输出文件名
            export_params=True,          # 是否在模型文件中存储权重
            opset_version=12,            # <--- 【关键修改】解决 unflatten 报错的核心
            do_constant_folding=True,    # 优化：预计算常量节点
            input_names=['src', 'tgt'],  # 给 Netron 里的输入节点起个易读的名字
            output_names=['output'],     # 给输出节点起名

            dynamic_axes={
                'src': {0: 'src_seq_len', 1: 'batch_size'},
                'tgt': {0: 'tgt_seq_len', 1: 'batch_size'},
                'output': {0: 'out_seq_len', 1: 'batch_size'}
            }
        )

if __name__ == "__main__":
    test()