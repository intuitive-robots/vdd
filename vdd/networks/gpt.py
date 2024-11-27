import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,

    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTNetwork(nn.Module):

    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 output_dim: int,
                 embed_dim: int,
                 embed_pdrop: float,
                 atten_pdrop: float,
                 resid_pdrop: float,
                 n_layers: int,
                 n_heads: int,
                 window_size: int,
                 goal_conditional: bool,
                 goal_seq_len: int = 1,
                 linear_output: bool = False,
                 pre_out_hidden_dim: int = 100,
                 encode_actions: bool = False,
                 action_dim: int = 0,
                 device: str = 'cuda', ):

        super(GPTNetwork, self).__init__()
        self.device = device
        self.goal_conditional = goal_conditional

        self.goal_seq_len = goal_seq_len
        if not goal_conditional:
            goal_dim = 0
            self.goal_seq_len = 0

        ### window size is only for the state sequence, by default only one goal and one readout token
        ### window size: for (state, action) pairs, the window size is 2 * window_size
        ### the goal sequence length is 1
        ### TODO: extend to multiple readout tokens
        if encode_actions:
            block_size = self.goal_seq_len + 2 * window_size
        else:
            block_size = self.goal_seq_len + window_size

        ### sequence size for the state sequence, every (state, action) pair at the same timestep share the same PE
        sequence_size = self.goal_seq_len + window_size

        ### output dim can be different to action dim,
        ### as we can predict means and cholenskys of all components
        self.out_dim = output_dim
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim

        self.encode_actions = encode_actions

        if encode_actions:
            self.action_dim = action_dim
            self.action_emb = nn.Linear(action_dim, embed_dim)

        # embedding layers
        ### Here we assume that the goal and state have the same dimension
        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, sequence_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrop)

        self.embed_dim = embed_dim

        self.block_size = block_size
        self.sequence_size = sequence_size
        self.window_size = window_size

        # transformer blocks
        self.blocks = nn.Sequential(
            *[Block(embed_dim, n_heads, atten_pdrop, resid_pdrop, block_size) for _ in range(n_layers)]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)

        if linear_output:
            self.head = nn.Linear(embed_dim, self.out_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, pre_out_hidden_dim),
                nn.SiLU(),
                nn.Linear(pre_out_hidden_dim, self.out_dim)
            )

        self.apply(self._init_weights)

        logger.info(f"Number of parameters in GPT: {sum(p.numel() for p in self.parameters())}")


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPTNetwork):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)


    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer


    def forward(self, states: torch.Tensor, goals: torch.Tensor = None, actions: torch.Tensor = None):
        """
        Run the model forward.
        states: (B, T, obs_dim)
        actions: (B, T-1, action_dim)
        goals: (B, T, goal_dim)
        """
        batch_size, window_size, dim = states.size()

        if self.encode_actions:
            assert actions.size(1) == window_size - 1, "Expected actions to have length T-1"

        assert window_size <= self.block_size, "Cannot forward, model block size is exhausted."
        assert dim == self.obs_dim, f"Expected state dim {self.obs_dim}, got {dim}"
        assert window_size <= self.window_size, f"Expected window size {self.window_size}, got {window_size}"

        state_embed = self.tok_emb(states)

        if self.goal_conditional:
            assert goals is not None, "Expected goals to be provided"
            assert goals.size(1) == self.goal_seq_len, f"Expected goal sequence length to be {self.goal_seq_len}, got {goals.size(1)}"
            goal_embed = self.tok_emb(goals)
            position_embeddings = self.pos_emb[:, :(window_size + self.goal_seq_len), :]
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        else:
            position_embeddings = self.pos_emb[:, :window_size, :]


        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])

        if self.encode_actions:
            action_embed = self.action_emb(actions)
            action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])

            sa_seq = torch.stack([state_x, action_x], dim=1
                                 ).permute(0, 2, 1, 3).reshape(batch_size, 2 * window_size, self.embed_dim)
        else:
            sa_seq = state_x

        # next we stack everything together
        if self.goal_conditional:
            input_seq = torch.cat([goal_x, sa_seq], dim=1)
        else:
            input_seq = sa_seq

        x = self.blocks(input_seq)
        x = self.ln_f(x)
        x = x[:, self.goal_seq_len:, :]

        assert x.shape[1] == states.shape[1], f"Expected output window size {states.shape[1]}, got {x.shape[1]}"

        out = self.head(x)

        return out

    def get_params(self):
        return self.parameters()