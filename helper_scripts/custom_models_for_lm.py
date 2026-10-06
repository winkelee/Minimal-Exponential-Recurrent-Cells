import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math


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
    def __init__(self, vocab_dim, hid=256, emb=64, device='cuda'):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.embedding = nn.Embedding(vocab_dim, emb, device=device)
        self.eGRUCell = eGRUCell(hid, emb, device)
        self.linear_pool = nn.Linear(hid, vocab_dim, device=device)
        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, start_state, prev_m, prev_norm, x):
        # x is of shape (batch, seq)

        batch_size = x.shape[0]
        logits_matrix = []

        #cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        #m = torch.zeros((batch_size, self.hid), device=self.device)
        #norm = torch.ones((batch_size, self.hid), device=self.device)

        cell_state = start_state
        m = prev_m
        norm = prev_norm

        emb_input = self.embedding(x) #(batch, seq, emb)

        for i in range(emb_input.shape[1]): #last entry is <STARTCOPY>
            current_token = emb_input[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.eGRUCell.forward(current_token, cell_state, m, norm)
            logits = self.linear_pool(hid_state)
            logits_matrix.append(logits)
        
        

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits, cell_state, m, norm


class LSTM_LM(nn.Module):
    def __init__(self, vocab_dim, hid=256, emb=64, device='cuda'):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.embedding = nn.Embedding(vocab_dim, emb, device=device)
        self.LSTMCell = nn.LSTMCell(emb, hid, device=device)
        self.linear_pool = nn.Linear(hid, vocab_dim, device=device)
        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, start_state, x):
        # x is of shape (batch, seq)

        batch_size = x.shape[0]
        logits_matrix = []

        #cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        #m = torch.zeros((batch_size, self.hid), device=self.device)
        #norm = torch.ones((batch_size, self.hid), device=self.device)

        hid_state = start_state
        cell_state = start_state

        emb_input = self.embedding(x) #(batch, seq, emb)

        for i in range(emb_input.shape[1]): #last entry is <STARTCOPY>
            current_token = emb_input[:, i] #(batch, dim)
            hid_state, cell_state = self.LSTMCell(current_token, (hid_state, cell_state))
            logits = self.linear_pool(hid_state)
            logits_matrix.append(logits)
        
        

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits, hid_state

class GRU_LM(nn.Module):
    def __init__(self, vocab_dim, hid=256, emb=64, device='cuda'):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.embedding = nn.Embedding(vocab_dim, emb, device=device)
        self.GRUCell = nn.GRUCell(emb, hid, device=device)
        self.linear_pool = nn.Linear(hid, vocab_dim, device=device)
        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, start_state, x):
        # x is of shape (batch, seq)

        batch_size = x.shape[0]
        logits_matrix = []

        #cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        #m = torch.zeros((batch_size, self.hid), device=self.device)
        #norm = torch.ones((batch_size, self.hid), device=self.device)

        hid_state = start_state

        emb_input = self.embedding(x) #(batch, seq, emb)

        for i in range(emb_input.shape[1]): #last entry is <STARTCOPY>
            current_token = emb_input[:, i] #(batch, dim)
            hid_state = self.GRUCell(current_token, hid_state)
            logits = self.linear_pool(hid_state)
            logits_matrix.append(logits)
        
        

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits, hid_state

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


    

    def forward(self, x, prev_cell_state, prev_hid_state, prev_m, prev_n, diagnostic=False):

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
    def __init__(self, vocab_dim, hid=256, emb=64, device='cuda'):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.embedding = nn.Embedding(vocab_dim, emb, device=device)
        self.sLSTMCell = sLSTMCell(hid, emb, device)
        self.linear_pool = nn.Linear(hid, vocab_dim, device=device)
        
        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, start_state, prev_m, prev_norm, x):
        # x is of shape (batch, seq)

        batch_size = x.shape[0]
        logits_matrix = []

        #cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        #m = torch.zeros((batch_size, self.hid), device=self.device)
        #norm = torch.ones((batch_size, self.hid), device=self.device)

        cell_state = start_state
        hid_state = start_state
        m = prev_m
        norm = prev_norm

        emb_input = self.embedding(x) #(batch, seq, emb)

        for i in range(emb_input.shape[1]): #last entry is <STARTCOPY>
            current_token = emb_input[:, i] #(batch, dim)
            cell_state, hid_state, m, norm = self.sLSTMCell.forward(current_token, cell_state, hid_state, m, norm)
            logits = self.linear_pool(hid_state)
            logits_matrix.append(logits)
        
        

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits, cell_state, m, norm

class MRRNCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, device=device)

        self.W = nn.Linear(emb, hid, device=device)
        self.U = nn.Linear(hid, hid, device=device)

    def forward(self, x, hid_state):


        z_tilde = self.W_z(x) + self.U_z(hid_state)
        h_tilde = self.W(x) + self.U(hid_state)

        z = F.sigmoid(z_tilde)
        h_tilde = F.tanh(h_tilde)

        new_hid_state = z * hid_state + (1-z)*h_tilde

        return new_hid_state

class MGU(nn.Module):
    def __init__(self, vocab_dim, hid=256, emb=64, device='cuda'):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.embedding = nn.Embedding(vocab_dim, emb, device=device)
        self.MGUCell = MRRNCell(hid, emb, device)
        self.linear_pool = nn.Linear(hid, vocab_dim, device=device)
        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, start_state, x):
        # x is of shape (batch, seq)

        batch_size = x.shape[0]
        logits_matrix = []

        #cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        #m = torch.zeros((batch_size, self.hid), device=self.device)
        #norm = torch.ones((batch_size, self.hid), device=self.device)

        hid_state = start_state

        emb_input = self.embedding(x) #(batch, seq, emb)

        for i in range(emb_input.shape[1]): #last entry is <STARTCOPY>
            current_token = emb_input[:, i] #(batch, dim)
            hid_state = self.MGUCell.forward(current_token, hid_state)
            logits = self.linear_pool(hid_state)
            logits_matrix.append(logits)
        
        

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits, hid_state

class eMGUCell(nn.Module):
    def __init__(self, hid, emb, device):
        super().__init__()

        self.hid = hid
        self.emb = emb

        self.W_z = nn.Linear(emb, hid, device=device)
        self.U_z = nn.Linear(hid, hid, device=device)

        self.W = nn.Linear(emb, hid, device=device)
        self.U = nn.Linear(hid, hid, device=device)

    def forward(self, x, hid_state, diagnostic=False):


        z_tilde = self.W_z(x) + self.U_z(hid_state)
        h_tilde = self.W(x) + self.U(hid_state)

        z = torch.exp(-F.softplus(z_tilde))
        h_tilde = F.tanh(h_tilde)

        new_hid_state = z * hid_state + (1-z)*h_tilde

        return new_hid_state

class eMGU(nn.Module):
    def __init__(self, vocab_dim, hid=256, emb=64, device='cuda'):
        super().__init__()

        self.hid = hid
        self.emb = emb
        self.device = device

        self.embedding = nn.Embedding(vocab_dim, emb, device=device)
        self.eMGUCell = eMGUCell(hid, emb, device)
        self.linear_pool = nn.Linear(hid, vocab_dim, device=device)
        self.init_hid = nn.Parameter(torch.randn(hid, device=device) * 0.1)

    def forward(self, start_state, x):
        # x is of shape (batch, seq)

        batch_size = x.shape[0]
        logits_matrix = []

        #cell_state = self.eGRUCell.init_hid.unsqueeze(0).expand(batch_size, -1)
        #m = torch.zeros((batch_size, self.hid), device=self.device)
        #norm = torch.ones((batch_size, self.hid), device=self.device)

        hid_state = start_state

        emb_input = self.embedding(x) #(batch, seq, emb)

        for i in range(emb_input.shape[1]): #last entry is <STARTCOPY>
            current_token = emb_input[:, i] #(batch, dim)
            hid_state = self.eMGUCell.forward(current_token, hid_state)
            logits = self.linear_pool(hid_state)
            logits_matrix.append(logits)
        
        

        final_logits = torch.stack(logits_matrix, dim=1)

        return final_logits, hid_state