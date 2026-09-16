from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import GINEConv, global_mean_pool

# Edge features: bond-type one-hot [4] + conjugation [1] + in-ring [1] + stereo one-hot [4]
# Node feature: create_graph --> [45].
EDGE_FEATURE_DIM = 10
NODE_FEATURE_DIM = 45


def _gine_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LeakyReLU(),
        nn.Linear(out_dim, out_dim),
    )


class GINEConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, feature_dim: int = NODE_FEATURE_DIM,
                 edge_dim: int = EDGE_FEATURE_DIM):
        super().__init__()
        self.output_dim = 2 * hidden_dim

        layer_dims = [feature_dim, hidden_dim, 2 * hidden_dim, 2 * hidden_dim, 2 * hidden_dim]
        self.gcn_layers = nn.ModuleList([
            GINEConv(_gine_mlp(layer_dims[i], layer_dims[i + 1]), edge_dim=edge_dim)
            for i in range(4)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
        ])
        self.gcn_act = nn.LeakyReLU()

    def forward(self, data_batch) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        )
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            x = gcn(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = self.gcn_act(x)
        return global_mean_pool(x, batch)


class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim: int, llm_dim: int, num_heads: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(graph_dim, llm_dim)
        self.attn = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(llm_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, graph_embedding: torch.Tensor, llm_hidden_states: torch.Tensor,
                llm_attention_mask: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(graph_embedding).unsqueeze(1) 
        key_padding_mask = ~llm_attention_mask.bool() 
        attn_out, _ = self.attn(query, llm_hidden_states, llm_hidden_states,
                                 key_padding_mask=key_padding_mask)
        fused = self.norm(self.out_proj(attn_out.squeeze(1)) + query.squeeze(1))
        return fused  


def build_shared_llm_backbone(llm_model_name: str, adapter_lora_configs: dict, torch_dtype=None):
    from transformers import AutoModel, AutoTokenizer
    from peft import get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_llm = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch_dtype)

    names = list(adapter_lora_configs.keys())
    peft_model = get_peft_model(base_llm, adapter_lora_configs[names[0]], adapter_name=names[0])
    for name in names[1:]:
        peft_model.add_adapter(name, adapter_lora_configs[name])

    return tokenizer, peft_model


def default_qwen2_lora_config(r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    from peft import LoraConfig

    return LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )


class _GNNLLMBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 256, feature_dim: int = NODE_FEATURE_DIM,
                 edge_dim: int = EDGE_FEATURE_DIM, use_llm: bool = False,
                 llm_model_name: Optional[str] = None, lora_config=None,
                 torch_dtype=None,
                 shared_llm_backbone=None, adapter_name: Optional[str] = None):
        super().__init__()
        self.use_llm = use_llm
        self.encoder = GINEConvEncoder(hidden_dim=hidden_dim, feature_dim=feature_dim, edge_dim=edge_dim)
        graph_dim = self.encoder.output_dim

        if use_llm:
            if shared_llm_backbone is not None:
                if adapter_name is None:
                    raise ValueError("adapter_name is required when shared_llm_backbone is provided")
                self.tokenizer, self.llm = shared_llm_backbone
                self._adapter_name = adapter_name
            else:
                if not llm_model_name:
                    raise ValueError("llm_model_name is required when use_llm=True and no shared_llm_backbone")
                from transformers import AutoModel, AutoTokenizer
                from peft import get_peft_model

                self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                base_llm = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch_dtype)
                self._adapter_name = adapter_name or "default"
                self.llm = (
                    get_peft_model(base_llm, lora_config, adapter_name=self._adapter_name)
                    if lora_config is not None else base_llm
                )
            llm_dim = self.llm.config.hidden_size
            self.fusion = CrossAttentionFusion(graph_dim, llm_dim)
            self.head_dim = llm_dim
        else:
            self.tokenizer = None
            self.llm = None
            self.fusion = None
            self.head_dim = graph_dim

        self.trunk = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.LeakyReLU(),
        )

    def _fused_features(self, data_batch, descriptor_texts: Optional[Sequence[str]]) -> torch.Tensor:
        graph_embedding = self.encoder(data_batch)

        if self.use_llm:
            if descriptor_texts is None:
                raise ValueError("descriptor_texts is required when use_llm=True")
            device = graph_embedding.device
            encoded = self.tokenizer(
                list(descriptor_texts), return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            llm_kwargs = dict(encoded)
            if hasattr(self.llm, "set_adapter"):
                llm_kwargs["adapter_names"] = [self._adapter_name] * encoded["input_ids"].shape[0]
            llm_out = self.llm(**llm_kwargs, output_hidden_states=True)
            hidden_states = getattr(llm_out, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = llm_out.hidden_states[-1]
            hidden_states = hidden_states.float()
            fused = self.fusion(graph_embedding, hidden_states, encoded["attention_mask"])
        else:
            fused = graph_embedding

        return self.trunk(fused)


class GNNLLMActor(_GNNLLMBackbone):
    def __init__(self, num_edit_ops: int, hidden_dim: int = 256,
                 feature_dim: int = NODE_FEATURE_DIM, edge_dim: int = EDGE_FEATURE_DIM,
                 use_llm: bool = False, llm_model_name: Optional[str] = None, lora_config=None,
                 torch_dtype=None,
                 shared_llm_backbone=None, adapter_name: Optional[str] = None,
                 edit_count_mode: str = "fixed", k_max: Optional[int] = None):
        super().__init__(hidden_dim=hidden_dim, feature_dim=feature_dim, edge_dim=edge_dim,
                          use_llm=use_llm, llm_model_name=llm_model_name, lora_config=lora_config,
                          torch_dtype=torch_dtype,
                          shared_llm_backbone=shared_llm_backbone, adapter_name=adapter_name)
        self.edit_count_mode = edit_count_mode
        self.k_max = k_max

        self.edit_head = nn.Linear(self.head_dim, num_edit_ops)
        if edit_count_mode == "learned":
            if not k_max or k_max < 1:
                raise ValueError("k_max must be a positive int when edit_count_mode == 'learned'")
            self.k_head = nn.Linear(self.head_dim, k_max)
        else:
            self.k_head = None

    def forward(self, data_batch, descriptor_texts: Optional[Sequence[str]] = None):
        features = self._fused_features(data_batch, descriptor_texts)
        edit_logits = self.edit_head(features)
        k_logits = self.k_head(features) if self.k_head is not None else None
        return edit_logits, k_logits


class GNNLLMCritic(_GNNLLMBackbone):
    OBJECTIVE_DIM = 4 

    def __init__(self, hidden_dim: int = 256, feature_dim: int = NODE_FEATURE_DIM,
                 edge_dim: int = EDGE_FEATURE_DIM, use_llm: bool = False,
                 llm_model_name: Optional[str] = None, lora_config=None,
                 torch_dtype=None,
                 shared_llm_backbone=None, adapter_name: Optional[str] = None):
        super().__init__(hidden_dim=hidden_dim, feature_dim=feature_dim, edge_dim=edge_dim,
                          use_llm=use_llm, llm_model_name=llm_model_name, lora_config=lora_config,
                          torch_dtype=torch_dtype,
                          shared_llm_backbone=shared_llm_backbone, adapter_name=adapter_name)
        self.value_head = nn.Linear(self.head_dim, self.OBJECTIVE_DIM)

    def forward(self, data_batch, descriptor_texts: Optional[Sequence[str]] = None) -> torch.Tensor:
        features = self._fused_features(data_batch, descriptor_texts)
        return self.value_head(features)  


def sample_edit(edit_logits: torch.Tensor):
    dist = Categorical(logits=edit_logits)
    idx = dist.sample()
    return idx, dist.log_prob(idx), dist.entropy()


def edit_log_prob_and_entropy(edit_logits: torch.Tensor, action_idx: torch.Tensor):
    dist = Categorical(logits=edit_logits)
    return dist.log_prob(action_idx), dist.entropy()
