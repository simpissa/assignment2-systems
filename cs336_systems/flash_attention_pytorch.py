import torch
from einops import einsum, rearrange
import math

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q = 8
        B_k = 8
        batch_dim, (N_k, d) = Q.shape[:-2], Q.shape[-2:]

        Q = rearrange(Q, "... (T_q B_q) d -> ... T_q B_q d", B_q=B_q) 
        K = rearrange(K, "... (T_k B_k) d -> ... T_k B_k d", B_k=B_k)
        V = rearrange(V, "... (T_k B_k) d -> ... T_k B_k d", B_k=B_k)

        T_q = Q.shape[-3]
        O = torch.empty((*batch_dim, N_k, d))
        L = torch.empty((*batch_dim, N_k))

        for i in range(T_q):
            Q_i = Q[..., i, :, :]
            O_i = torch.zeros((*batch_dim, B_q, d))
            l = torch.zeros((*batch_dim, B_q))
            m = torch.full((*batch_dim, B_q), float('-inf'))

            for j in range(K.shape[-3]):
                prev_m = m
                K_j = K[..., j, :, :]
                V_j = V[..., j, :, :]
                S = einsum(Q_i, K_j, "... B_q d, ... B_k d -> ... B_q B_k") / math.sqrt(d)
                m = torch.maximum(m, torch.max(S, dim=-1).values)
                P = torch.exp(S - m.unsqueeze(-1))
                l = torch.exp(prev_m - m) * l + torch.sum(P, dim=-1)
                O_i = torch.exp(prev_m - m).unsqueeze(-1) * O_i + einsum(P, V_j, "... B_q B_k, ... B_k d -> ... B_q d")
            
            O_i = O_i / l.unsqueeze(-1)
            L_i = m + torch.log(l)

            O[:, i*B_q:(i+1)*B_q, :] = O_i
            L[:, i*B_q:(i+1)*B_q] = L_i

        Q = rearrange(Q, "... T_q B_q d -> ... (T_q B_q) d", B_q=B_q) 
        K = rearrange(K, "... T_k B_k d -> ... (T_k B_k) d", B_k=B_k)
        V = rearrange(V, "... T_k B_k d -> ... (T_k B_k) d", B_k=B_k)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.d = d
        return O
        
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        D = torch.sum(O * dO, dim=-1, keepdim=True)
        S = Q @ K.transpose(-1, -2) / math.sqrt(ctx.d)
        P = torch.exp(S - L.unsqueeze(-1))
        dV = P.transpose(-1, -2) @ dO
        dP = dO @ V.transpose(-1, -2)
        dS = P * (dP - D)
        dQ = dS @ K / math.sqrt(ctx.d)
        dK = dS.transpose(-1, -2) @ Q / math.sqrt(ctx.d)

        return dQ, dK, dV, None
    
