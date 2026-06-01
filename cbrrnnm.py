import torch
from cbrrnn import CbrRnnCell, CbrRnnJITLoop, CbrRnn
from typing import List


class CbrRnnMCell(CbrRnnCell):
    def __init__(self, d_model, d_qk, d_v, d_hid, update_weight):
        super().__init__(d_model, d_qk, d_v, d_hid)
        self.update_weight = update_weight

    def attend(self, q_t: torch.Tensor, kv_t: torch.Tensor) -> torch.Tensor:
        # q_t: [batch, d_qk]
        # kv_t: [batch, d_v, d_qk]
        q_t = q_t.unsqueeze(-1) # [batch, d_qk, 1]
        c_t = torch.matmul(kv_t, q_t).squeeze(-1) # [batch, d_v, 1] -> [batch, d_v]
        return c_t

    def compute_mem_update(self, *args):
        raise AttributeError("no standalone memory update calculation for CBR-RNN-M")

    def update_kv(self, kv_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor):
        # kv_t: [batch, d_v, d_qk]
        # k_t: [batch, d_qk]
        # v_t: [batch, d_v]
        # predicted value
        v_t_pred = torch.einsum('b v d, b d -> b v', kv_t, k_t) # [batch, d_v]
        # difference between predicted value and actual value
        v_diff = v_t - v_t_pred # [batch, d_v]
        kv_diff = v_diff.unsqueeze(2) * k_t.unsqueeze(1) # [batch, d_v, d_k]
        mem_update = torch.linalg.matrix_norm(kv_diff) # [batch]
        # updated associative memory
        kv_t = kv_t + self.update_weight*kv_diff
        return kv_t, mem_update


class CbrRnnMJITLoop(CbrRnnJITLoop):
    def forward(self, inputs: torch.Tensor, kv_init: torch.Tensor, h_init: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # inputs: [seq_len, batch, d_model]
        #batch_size = inputs.size(1)
        
        hiddens = torch.jit.annotate(List[torch.Tensor], [])
        mem_updates = torch.jit.annotate(List[torch.Tensor], [])

        h_t = h_init # [batch, d_hid]
        kv_t = kv_init # [batch, d_v, d_qk]

        for i in range(inputs.size(0)):
            x_t = inputs[i] # [batch, d_model]
            
            q_input = torch.concat([x_t, h_t], dim=1) # [batch, d_model+d_hid]
            q_raw = self.layer.weight_q(q_input) # [batch, d_qk]
            # tanh activation and layer normalization are not applied
            # to query, key, and value
            q_t = self.layer.apply_rope(q_raw, cos[i], sin[i]) # [batch, d_qk]
            c_t = self.layer.attend(q_t, kv_t)
            inter_input = torch.concat([x_t, c_t, q_t, h_t], dim=1) # [batch, d_inter]
            inter = self.layer.weight_inter(inter_input) # [batch, d_inter]
            inter = self.layer.inter_norm(inter)
            inter_output = self.layer.tanh(inter)
            k_raw = self.layer.weight_k(inter_output) # [batch, d_qk]
            k_t = self.layer.apply_rope(k_raw, cos[i], sin[i]) # [batch, d_qk]
            v_t = self.layer.weight_v(inter_output) # [batch, d_v]
            kv_t, mem_update = self.layer.update_kv(kv_t, k_t, v_t)
            
            # compute new hidden state
            hid = self.layer.weight_h(inter_output) # [batch, d_hid]
            hid = self.layer.h_norm(hid)
            h_t = self.layer.tanh(hid)
            hiddens.append(h_t)
            mem_updates.append(mem_update)
        return torch.stack(hiddens), torch.stack(mem_updates) # [seq_len, batch, d_hid], [seq_len, batch]


class CbrRnnM(CbrRnn):
    def __init__(self, vocab_size, embed_dim, d_qk, d_v, d_hid, max_seq_len, rope_base, update_weight):
        super().__init__(vocab_size, embed_dim, d_qk, d_v, d_hid, max_seq_len, rope_base)
        self.update_weight = update_weight
        self.rnn = torch.jit.script(CbrRnnMJITLoop(CbrRnnMCell(
            embed_dim, d_qk, d_v, d_hid, update_weight
        ))) 

    def forward(self, x, hidden=None, kv=None):
        # x: [batch, seq_len] -> [seq_len, batch]
        x = x.t() 
        seq_len = x.size(0)
        batch_size = x.size(1)
        
        x_emb = self.embedding(x)

        # Initialize associative memory and hidden state if none provided
        if kv is None:
            kv = torch.zeros(batch_size, self.d_v, self.d_qk, device=x.device)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.d_hid, device=x.device)

        # concatenate embedding with hidden state
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        
        hiddens, mem_updates = self.rnn(x_emb, kv, hidden, cos, sin)
        rnn_out = hiddens.transpose(0, 1) # [batch, seq_len, d_hid]
        logits = self.decoder(rnn_out)
        
        return logits, hiddens, mem_updates
