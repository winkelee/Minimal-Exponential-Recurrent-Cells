import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math


class mGRUCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_r1 = nn.Parameter(torch.randn(hid, emb, device=device) * 0.1)
        self.W_r2 = nn.Parameter(torch.randn(hid, emb, device=device) * 0.1)
        self.U_r = nn.Parameter(torch.randn(hid, hid, device=device) * 0.1)
        self.B_wr = nn.Parameter(torch.ones(hid, hid, device=device) * 3.0)
        self.B_hr = nn.Parameter(torch.ones(hid, hid, device=device) * 3.0)

        self.W_z1 = nn.Parameter(torch.randn(hid, emb, device=device) * 0.1)
        self.W_z2 = nn.Parameter(torch.randn(hid, emb, device=device) * 0.1)
        self.U_z = nn.Parameter(torch.randn(hid, hid, device=device) * 0.1)
        self.B_wz = nn.Parameter(torch.ones(hid, hid, device=device) * 3.0)
        self.B_hz = nn.Parameter(torch.ones(hid, hid, device=device) * 3.0)

        self.W_1 = nn.Parameter(torch.randn(hid, emb, device=device) * 0.1)
        self.W_2 = nn.Parameter(torch.randn(hid, emb, device=device) * 0.1)
        self.U = nn.Parameter(torch.randn(hid, hid, device=device) * 0.1)
        self.B_wn = nn.Parameter(torch.ones(hid, hid, device=device) * 3.0)
        self.B_hn = nn.Parameter(torch.ones(hid, hid, device=device) * 3.0)

        self.query_linear = nn.Linear(emb, hid, bias=False, device=device)

        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, hid, device=device) * 0.1)

    def forward(self, x, prev_hid_state):

        v_r1 = F.linear(x, self.W_r1) # (batch_size, hid)
        v_r2 = F.linear(x, self.W_r2) # (batch_size, hid)

        Wr_tilde = torch.einsum('bi,bj->bij', v_r1, v_r2) + self.B_wr
        w_H_r = torch.einsum('ij,bjk->bik', self.U_r, prev_hid_state) + self.B_hr 
        R = torch.sigmoid(Wr_tilde + w_H_r)
        

        v_z1 = F.linear(x, self.W_z1)
        v_z2 = F.linear(x, self.W_z2)
        Wz_tilde = torch.einsum('bi,bj->bij', v_z1, v_z2) + self.B_wz
        w_H_z = torch.einsum('ij,bjk->bik', self.U_z, prev_hid_state) + self.B_hz
        Z = torch.sigmoid(Wz_tilde + w_H_z)

        v_1 = F.linear(x, self.W_1)
        v_2 = F.linear(x, self.W_2)
        W_tilde = torch.einsum('bi,bj->bij', v_1, v_2) + self.B_wn
        w_H = torch.einsum('ij,bjk->bik', self.U, R * prev_hid_state) + self.B_hn
        H_tilde = F.tanh(W_tilde + w_H)

        new_hidden = Z * prev_hid_state + (1 - Z) * H_tilde

        return new_hidden

class mGRU(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.mGRUCell = mGRUCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        state = self.mGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1, -1)
        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            state = self.mGRUCell.forward(current_token, state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        state = self.mGRUCell.forward(token, state)
        q = self.mGRUCell.query_linear(token)
        vector_state = torch.einsum('bjk,bk->bj', state, q)
        logits = self.mGRUCell.linear_pool(vector_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            state = self.mGRUCell.forward(token, state)
            q = self.mGRUCell.query_linear(token)
            vector_state = torch.einsum('bjk,bk->bj', state, q)
            logits = self.mGRUCell.linear_pool(vector_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        state = self.mGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1, -1)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            state = self.mGRUCell.forward(current_token, state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        state = self.mGRUCell.forward(token, state)
        q = self.mGRUCell.query_linear(token)
        vector_state = torch.einsum('bjk,bk->bj', state, q)
        logits = self.mGRUCell.linear_pool(vector_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            state = self.mGRUCell.forward(token, state)
            q = self.mGRUCell.query_linear(token)
            vector_state = torch.einsum('bjk,bk->bj', state, q)
            logits = self.mGRUCell.linear_pool(vector_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits


class CopyGRU(nn.Module): # To bench mGRU against
    def __init__(self, hid, emb, device="cuda"):
        super().__init__()

        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)
        self.GRUCell = nn.GRUCell(emb, hid, device=device)
        self.linear_proj = nn.Linear(hid, emb, device=device)

    def forward(self, x, y):

        batch_size = x.shape[0]
        state = self.init_hid.unsqueeze(0).expand(batch_size, -1)
        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            state = self.GRUCell(current_token, state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        state = self.GRUCell(token, state)
        logits = self.linear_proj(state) #(batch, emb)
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1):
            token = y[:, i]
            state = self.GRUCell(token, state)
            logits = self.linear_proj(state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    
    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        state = self.init_hid.unsqueeze(0).expand(batch_size, -1)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            state = self.GRUCell(current_token, state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        state = self.GRUCell(token, state)
        logits = self.linear_proj(state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
        step=0

        while step < max_len:
            state = self.GRUCell(token, state)
            logits = self.linear_proj(state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step = step + 1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits


class mLSTMLiteCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_q = nn.Linear(emb, hid, device=device)
        self.W_k = nn.Linear(emb, hid, device=device)
        self.W_v = nn.Linear(emb, hid, device=device)

        self.w_i = nn.Linear(emb, 1, device=device)
        self.w_f = nn.Linear(emb, 1, device=device)

        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, hid, device=device) * 0.1)

    def forward(self, x, prev_cell_state, prev_m, prev_n):

        q = self.W_q(x) #(batch, hid)
        k = self.W_k(x)/ math.sqrt(self.hid) #(batch, hid)
        v = self.W_v(x) #(batch, hid)
        i_tilde = self.w_i(x) #(batch, 1)
        f_tilde = self.w_f(x) #(batch, 1)

        m = torch.max(f_tilde + prev_m, i_tilde) #(batch, 1)

        f = torch.exp(f_tilde + prev_m - m) #(batch, 1)
        i = torch.exp(i_tilde - m) #(batch, 1)

        f_matrix = f.unsqueeze(-1)
        i_matrix = i.unsqueeze(-1)

        #prev_norm = (batch, hid)

        new_norm = f*prev_n + i*k #(batch, hid)
        outer_vk = torch.einsum('bi,bj->bij', v, k)  #(batch, hid, hid) ?
        cell_state = f_matrix*prev_cell_state + i_matrix*outer_vk #(batch, hid, hid)

        
        dot_nq = torch.sum(new_norm * q, dim=-1, keepdim=True)
        
        denom = torch.clamp(torch.abs(dot_nq), min=1.0)
        
        numerator = torch.einsum('bjk,bk->bj', cell_state, q)
        
        h_tilde = numerator / denom

        new_hid_state = h_tilde

        return cell_state, new_hid_state, m, new_norm

class mLSTMLite(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.mLSTMLiteCell = mLSTMLiteCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        cell_state = self.mLSTMLiteCell.init_hid.unsqueeze(0).expand(batch_size, -1, -1)
        logits_matrix = []

         
        m = torch.zeros((batch_size, 1), device=self.device)
        norm = torch.ones((batch_size, self.hid), device=self.device)

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.mLSTMLiteCell.forward(
                x=current_token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.mLSTMLiteCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        logits = self.mLSTMLiteCell.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            cell_state, hid_state, m, norm = self.mLSTMLiteCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
            logits = self.mLSTMLiteCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        cell_state = self.mLSTMLiteCell.init_hid.unsqueeze(0).expand(batch_size, -1, -1)

         
        m = torch.zeros((batch_size, 1), device=self.device)
        norm = torch.ones((batch_size, self.hid), device=self.device)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.mLSTMLiteCell.forward(
                x=current_token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.mLSTMLiteCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        logits = self.mLSTMLiteCell.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            cell_state, hid_state, m, norm = self.mLSTMLiteCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
            logits = self.mLSTMLiteCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

class mLSTMCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_q = nn.Linear(emb, hid, device=device)
        self.W_k = nn.Linear(emb, hid, device=device)
        self.W_v = nn.Linear(emb, hid, device=device)

        self.w_i = nn.Linear(emb, 1, device=device)
        self.w_f = nn.Linear(emb, 1, device=device)

        self.W_o = nn.Linear(emb, hid, device=device)

        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, hid, device=device) * 0.1)

    def forward(self, x, prev_cell_state, prev_m, prev_n):

        q = self.W_q(x) #(batch, hid)
        k = self.W_k(x)/ math.sqrt(self.hid) #(batch, hid)
        v = self.W_v(x) #(batch, hid)
        i_tilde = self.w_i(x) #(batch, 1)
        f_tilde = self.w_f(x) #(batch, 1)
        o_tilde = self.W_o(x) #(batch, hid)

        o = F.sigmoid(o_tilde) #(batch, hid)

        m = torch.max(f_tilde + prev_m, i_tilde) #(batch, 1)

        f = torch.exp(f_tilde + prev_m - m) #(batch, 1)
        i = torch.exp(i_tilde - m) #(batch, 1)

        f_matrix = f.unsqueeze(-1)
        i_matrix = i.unsqueeze(-1)

        #prev_norm = (batch, hid)

        new_norm = f*prev_n + i*k #(batch, hid)
        outer_vk = torch.einsum('bi,bj->bij', v, k)  #(batch, hid, hid) ?
        cell_state = f_matrix*prev_cell_state + i_matrix*outer_vk #(batch, hid, hid)

        
        dot_nq = torch.sum(new_norm * q, dim=-1, keepdim=True)
        
        denom = torch.clamp(torch.abs(dot_nq), min=1.0)
        
        numerator = torch.einsum('bjk,bk->bj', cell_state, q)
        
        h_tilde = numerator / denom

        new_hid_state = o * h_tilde

        return cell_state, new_hid_state, m, new_norm

class mLSTM(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.mLSTMCell = mLSTMCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        cell_state = self.mLSTMCell.init_hid.unsqueeze(0).expand(batch_size, -1, -1)
        logits_matrix = []

         
        m = torch.zeros((batch_size, 1), device=self.device)
        norm = torch.ones((batch_size, self.hid), device=self.device)

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.mLSTMCell.forward(
                x=current_token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.mLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        logits = self.mLSTMCell.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            cell_state, hid_state, m, norm = self.mLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
            logits = self.mLSTMCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        cell_state = self.mLSTMCell.init_hid.unsqueeze(0).expand(batch_size, -1, -1)

         
        m = torch.zeros((batch_size, 1), device=self.device)
        norm = torch.ones((batch_size, self.hid), device=self.device)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.mLSTMCell.forward(
                x=current_token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.mLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
        logits = self.mLSTMCell.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            cell_state, hid_state, m, norm = self.mLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_m=m,
                prev_n=norm)
            logits = self.mLSTMCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits


class sLSTMCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, bias=False, device=device)

        self.W_i = nn.Linear(emb, hid, device=device)
        self.U_i = nn.Linear(hid, hid, bias=False, device=device)

        self.W_o = nn.Linear(emb, hid, device=device)
        self.U_o = nn.Linear(hid, hid, bias=False, device=device)

        self.W_f = nn.Linear(emb, hid, device=device)
        self.U_f = nn.Linear(hid, hid, bias=False, device=device)


        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, x, prev_cell_state, prev_hid_state, prev_m, prev_n):

        z_tilde = self.W_z(x) + self.U_z(prev_hid_state)
        f_tilde = self.W_f(x) + self.U_f(prev_hid_state)
        i_tilde = self.W_i(x) + self.U_i(prev_hid_state)
        o_tilde = self.W_o(x) + self.U_o(prev_hid_state)

        o = F.sigmoid(o_tilde)
        z = F.tanh(z_tilde)

        m = torch.max(f_tilde + prev_m, i_tilde)

        f = torch.exp(f_tilde + prev_m - m)
        i = torch.exp(i_tilde - m)

        new_norm = f*prev_n + i

        cell_state = f*prev_cell_state + i*z

        h_tilde = cell_state / new_norm

        new_hid_state = o * h_tilde

        return cell_state, new_hid_state, m, new_norm

class sLSTM(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.sLSTMCell = sLSTMCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        cell_state = self.sLSTMCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        logits_matrix = []

         
        m = torch.zeros((batch_size, self.hid), device=self.device)
        norm = torch.ones((batch_size, self.hid), device=self.device)
        hid_state = cell_state / norm 

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.sLSTMCell.forward(
                x=current_token, 
                prev_cell_state=cell_state, 
                prev_hid_state=hid_state,
                prev_m=m,
                prev_n=norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.sLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_hid_state=hid_state,
                prev_m=m,
                prev_n=norm)
        logits = self.sLSTMCell.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            cell_state, hid_state, m, norm = self.sLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_hid_state=hid_state,
                prev_m=m,
                prev_n=norm)
            logits = self.sLSTMCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        cell_state = self.sLSTMCell.init_hid.unsqueeze(0).expand(batch_size, -1)

         
        m = torch.zeros((batch_size, self.hid), device=self.device)
        norm = torch.ones((batch_size, self.hid), device=self.device)
        hid_state = cell_state / norm 

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.sLSTMCell.forward(
                x=current_token, 
                prev_cell_state=cell_state, 
                prev_hid_state=hid_state,
                prev_m=m,
                prev_n=norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.sLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_hid_state=hid_state,
                prev_m=m,
                prev_n=norm)
        logits = self.sLSTMCell.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            cell_state, hid_state, m, norm = self.sLSTMCell.forward(
                x=token, 
                prev_cell_state=cell_state, 
                prev_hid_state=hid_state,
                prev_m=m,
                prev_n=norm)
            logits = self.sLSTMCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits



class eGRUCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, device=device)

        self.W_r = nn.Linear(emb, hid, device=device)
        self.U_r = nn.Linear(hid, hid, device=device)

        self.W = nn.Linear(emb, hid, device=device)
        self.U = nn.Linear(hid, hid, device=device)


        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, x, prev_cell_state, prev_m, prev_n):

        hid_state = prev_cell_state / prev_n

        z_tilde = self.W_z(x) + self.U_z(hid_state)
        r_tilde = self.W_r(x) + self.U_r(hid_state)
        h_tilde = self.W(x) + self.U(hid_state)
        h_tilde = F.tanh(h_tilde)

        m = torch.max(z_tilde + prev_m, r_tilde)

        z = torch.exp(z_tilde + prev_m - m)
        r = torch.exp(r_tilde - m)

        new_norm = z*prev_n + r

        cell_state = z*prev_cell_state + r*h_tilde
        new_hid_state = cell_state / new_norm

        return cell_state, new_hid_state, m, new_norm

class eGRU(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.eGRUCell = eGRUCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        logits_matrix = []

        m = torch.zeros((batch_size, self.hid), device=self.device)
        norm = torch.ones((batch_size, self.hid), device=self.device)

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.eGRUCell.forward(current_token, cell_state, m, norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.eGRUCell.forward(token, cell_state, m, norm)
        logits = self.eGRUCell.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            cell_state, hid_state, m, norm = self.eGRUCell.forward(token, cell_state, m, norm)
            logits = self.eGRUCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.eGRUCell.forward(current_token, cell_state, m, norm)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        cell_state, hid_state, m, norm = self.eGRUCell.forward(token, cell_state, m, norm)
        logits = self.eGRUCell.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            cell_state, hid_state, m, norm = self.eGRUCell.forward(token, cell_state, m, norm)
            logits = self.eGRUCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits


class deGRUCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, device=device)

        self.W_r = nn.Linear(emb, hid, device=device)
        self.U_r = nn.Linear(hid, hid, device=device)

        self.W = nn.Linear(emb, hid, device=device)
        self.U = nn.Linear(hid, hid, device=device)


        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, x, hid_state):


        z_tilde = self.W_z(x) + self.U_z(hid_state)
        r_tilde = self.W_r(x) + self.U_r(hid_state)
        h_tilde = self.W(x) + self.U(hid_state)

        z = F.sigmoid(z_tilde)
        r = F.sigmoid(r_tilde)
        h_tilde = F.tanh(h_tilde)

        new_hid_state = z * hid_state + r*h_tilde

    

        return new_hid_state

class deGRU(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.deGRUCell = deGRUCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        hid_state = self.deGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.deGRUCell.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.deGRUCell.forward(token, hid_state)
        logits = self.deGRUCell.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            hid_state = self.deGRUCell.forward(token, hid_state)
            logits = self.deGRUCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        cell_state = self.deGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.deGRUCell.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.deGRUCell.forward(token, hid_state)
        logits = self.deGRUCell.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            hid_state = self.deGRUCell.forward(token, hid_state)
            logits = self.deGRUCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits


class eGRUCell_rmsn(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, device=device)

        self.W_r = nn.Linear(emb, hid, device=device)
        self.U_r = nn.Linear(hid, hid, device=device)

        self.W = nn.Linear(emb, hid, device=device)
        self.U = nn.Linear(hid, hid, device=device)

        self.rmsn_norm = nn.RMSNorm(hid, device=device)

        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, x, hid_state):


        z_tilde = self.W_z(x) + self.U_z(hid_state)
        r_tilde = self.W_r(x) + self.U_r(hid_state)
        h_tilde = self.W(x) + self.U(hid_state)
        h_tilde = F.tanh(h_tilde)

        z = F.relu(z_tilde)
        r = F.relu(r_tilde)

        raw_hid = z * hid_state + r * h_tilde

        new_hid_state = self.rmsn_norm(raw_hid)


        

        return new_hid_state

class eGRU_rmsn(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.eGRUCell_rmsn = eGRUCell_rmsn(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        hid_state = self.eGRUCell_rmsn.init_hid.unsqueeze(0).expand(batch_size, -1)
        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.eGRUCell_rmsn.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.eGRUCell_rmsn.forward(token, hid_state)
        logits = self.eGRUCell_rmsn.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            hid_state = self.eGRUCell_rmsn.forward(token, hid_state)
            logits = self.eGRUCell_rmsn.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        hid_state = self.eGRUCell_rmsn.init_hid.unsqueeze(0).expand(batch_size, -1)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.eGRUCell_rmsn.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.eGRUCell_rmsn.forward(token, hid_state)
        logits = self.eGRUCell_rmsn.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            hid_state = self.eGRUCell_rmsn.forward(token, hid_state)
            logits = self.eGRUCell_rmsn.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

class MRRNCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, device=device)

        self.W = nn.Linear(emb, hid, device=device)
        self.U = nn.Linear(hid, hid, device=device)


        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, x, hid_state):


        z_tilde = self.W_z(x) + self.U_z(hid_state)
        h_tilde = self.W(x) + self.U(hid_state)

        z = F.sigmoid(z_tilde)
        h_tilde = F.tanh(h_tilde)

        new_hid_state = z * hid_state + (1-z)*h_tilde

    

        return new_hid_state

class MRRN(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.MRRNCell = MRRNCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        hid_state = self.MRRNCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.MRRNCell.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.MRRNCell.forward(token, hid_state)
        logits = self.MRRNCell.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            hid_state = self.MRRNCell.forward(token, hid_state)
            logits = self.MRRNCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        hid_state = self.MRRNCell.init_hid.unsqueeze(0).expand(batch_size, -1)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.MRRNCell.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.MRRNCell.forward(token, hid_state)
        logits = self.MRRNCell.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            hid_state = self.MRRNCell.forward(token, hid_state)
            logits = self.MRRNCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits


class maGRUCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, device=device)

        self.W = nn.Linear(emb, hid, device=device)
        self.U = nn.Linear(hid, hid, device=device)


        self.linear_pool = nn.Linear(hid, emb, device=device)

        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, x, hid_state):


        z_tilde = self.W_z(x) + self.U_z(hid_state)
        h_tilde = self.W(x) + self.U(hid_state)

        z = torch.exp(-F.softplus(z_tilde))
        h_tilde = F.tanh(h_tilde)

        new_hid_state = z * hid_state + (1-z)*h_tilde

    

        return new_hid_state

class maGRU(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.maGRUCell = maGRUCell(hid, emb, device)

    def forward(self, x, y):
        # x is of shape (batch, seq, emb)

        batch_size = x.shape[0]
        hid_state = self.maGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.maGRUCell.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.maGRUCell.forward(token, hid_state)
        logits = self.maGRUCell.linear_pool(hid_state) #(batch, emb) FIRST NUMBER PREDICTION
        logits_matrix.append(logits)

        for i in range(y.shape[1] - 1): #Predict the rest of n-1 numbers
            token = y[:, i]
            hid_state = self.maGRUCell.forward(token, hid_state)
            logits = self.maGRUCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits

    def predict_inference(self, x, vocab, max_len=30):

        batch_size = x.shape[0]
        hid_state = self.maGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)

        logits_matrix = []

        for i in range(x.shape[1]-1): #last entry is <STARTCOPY>
            current_token = x[:, i] #(batch, dim)
            hid_state = self.maGRUCell.forward(current_token, hid_state)
        
        # The decoding starts here

        token = x[:, x.shape[1]-1] #load the <STARTCOPY> entry
        hid_state = self.maGRUCell.forward(token, hid_state)
        logits = self.maGRUCell.linear_pool(hid_state) #(batch, emb)
        logits_matrix.append(logits)
        token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float() 
        step=0

        while step < max_len:
            hid_state = self.maGRUCell.forward(token, hid_state)
            logits = self.maGRUCell.linear_pool(hid_state) #(batch, emb)
            logits_matrix.append(logits)
            token = F.one_hot(torch.argmax(logits, dim=1), num_classes=len(vocab)).float()
            step= step +1

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits
