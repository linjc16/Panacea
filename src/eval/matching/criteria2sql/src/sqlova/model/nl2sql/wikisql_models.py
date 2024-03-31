# Copyright 2019-present NAVER Corp.
# Apache License v2.0
# Wonseok modified. 20180607

# Xiaojing modified, 20190501

import os, json
from copy import deepcopy
from matplotlib.pylab import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sqlova.utils.utils import topk_multi_dim
from sqlova.utils.utils_wikisql import *

class Seq2SQL_v1(nn.Module):
    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, n_comb_ops, old=False):
        super(Seq2SQL_v1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr

        self.max_wn = 20
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_comb_ops = n_comb_ops

        self.wcp = WCP(iS, hS, lS, dr)
        self.scp = SCP(iS, hS, lS, dr)
        self.sap = SAP(iS, hS, lS, dr, n_agg_ops, old=old)
        self.wnp = WNP(iS, hS, lS, dr)
        self.wcp = WCP(iS, hS, lS, dr)
        self.wop = WOP(iS, hS, lS, dr, n_cond_ops)
        self.wvp = WVP_se(iS, hS, lS, dr, n_cond_ops, old=old) # start-end-search-discriminative model
        self.wbool = WBOOL(iS, hS, lS, dr)
        self.wao = WAO(iS, hS, lS, dr, n_comb_ops)
        self.word = WORD(iS, hS, lS, dr, n_comb_ops)

        #iS:

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs,
                g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None, g_ao=None, g_ord=None,
                show_p_sc=False, show_p_sa=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False,show_p_wb=False, show_p_ao=False, show_p_ord=False ,hs_type=None):

        # sc
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)

        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)

        # sa
        s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa)
        if g_sa:
            # it's not necessary though.
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)


        # wn
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)

        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)

        # wc

        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, show_p_wc=show_p_wc, penalty=True)

        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)

        # wo
        s_wo = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_wo=show_p_wo)

        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)

        #ao
        s_ao = self.wao(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_ao=show_p_ao)
        if g_ao:
            pr_ao = g_ao
        else:
            pr_ao = pred_ao(pr_wn, s_ao)

        #ord
        s_ord = self.word(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_ord=show_p_ord)

        if g_ord:
            pr_ord = g_ord
        else:
            pr_ord = pred_ord(pr_wn, s_ord)

        # wv
        s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv)

        s_wb = self.wbool(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wb=show_p_wb)

        s_type = torch.cat([s_wv, s_wb.unsqueeze(2)], dim=2)

        return s_sc, s_sa, s_wn, s_wc, s_wo, s_type, s_ao, s_ord


class SCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(SCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr


        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=False):
        # Encode

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)

        #   [bS, mL_hs, 100] * [bS, 100, mL_n] -> [bS, mL_hs, mL_n]
        att_h = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))

        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_h[b, :, l_n1:] = -10000000000

        p_n = self.softmax_dim2(att_h)
        if show_p_sc:
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001, figsize=(12,3.5))
            subplot2grid((7,2), (3, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p_n[0][i_h][:].data.numpy() - i_h, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('sc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        #   p_n [ bS, mL_hs, mL_n]  -> [ bS, mL_hs, mL_n, 1]
        #   wenc_n [ bS, mL_n, 100] -> [ bS, 1, mL_n, 100]
        #   -> [bS, mL_hs, mL_n, 100] -> [bS, mL_hs, 100]
        c_n = torch.mul(p_n.unsqueeze(3), wenc_n.unsqueeze(1)).sum(dim=2)

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)
        s_sc = self.sc_out(vec).squeeze(2) # [bS, mL_hs, 1] -> [bS, mL_hs]

        # Penalty
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                s_sc[b, l_hs1:] = -10000000000

        return s_sc


class SAP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_agg_ops=-1, old=False):
        super(SAP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr


        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.sa_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, n_agg_ops))  # Fixed number of aggregation operator.

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

        if old:
            # for backwoard compatibility
            self.W_c = nn.Linear(hS, hS)
            self.W_hs = nn.Linear(hS, hS)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=False):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)

        wenc_hs_ob = wenc_hs[list(range(bS)), pr_sc]  # list, so one sample for each batch.

        # [bS, mL_n, 100] * [bS, 100, 1] -> [bS, mL_n]
        att = torch.bmm(self.W_att(wenc_n), wenc_hs_ob.unsqueeze(2)).squeeze(2)

        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, l_n1:] = -10000000000
        # [bS, mL_n]
        p = self.softmax_dim1(att)

        if show_p_sa:
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,3)
            cla()
            plot(p[0].data.numpy(), '--rs', ms=7)
            title('sa: nlu_weight')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        #    [bS, mL_n, 100] * ( [bS, mL_n, 1] -> [bS, mL_n, 100])
        #       -> [bS, mL_n, 100] -> [bS, 100]
        c_n = torch.mul(wenc_n, p.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_sa = self.sa_out(c_n)


        return s_sa


class WNP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, ):
        super(WNP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 20  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att_h = nn.Linear(hS, 1)
        self.W_hidden = nn.Linear(hS, lS * hS)
        self.W_cell = nn.Linear(hS, lS * hS)

        self.W_att_n = nn.Linear(hS, 1)
        self.wn_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, self.mL_w + 1))  # max number (4 + 1)

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=False):
        # Encode

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, mL_hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)
        mL_hs = max(l_hs)
        # mL_h = max(l_hpu)

        #   (self-attention?) column Embedding?
        #   [B, mL_hs, 100] -> [B, mL_hs, 1] -> [B, mL_hs]
        att_h = self.W_att_h(wenc_hs).squeeze(2)

        #   Penalty
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                att_h[b, l_hs1:] = -10000000000
        p_h = self.softmax_dim1(att_h)

        if show_p_wn:
            if p_h.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,5)
            cla()
            plot(p_h[0].data.numpy(), '--rs', ms=7)
            title('wn: header_weight')
            grid(True)
            fig.canvas.draw()
            show()
            # input('Type Eenter to continue.')

        #   [B, mL_hs, 100] * [ B, mL_hs, 1] -> [B, mL_hs, 100] -> [B, 100]
        c_hs = torch.mul(wenc_hs, p_h.unsqueeze(2)).sum(1)

        #   [B, 100] --> [B, 2*100] Enlarge because there are two layers.
        hidden = self.W_hidden(c_hs)  # [B, 4, 200/2]
        hidden = hidden.view(bS, self.lS * 2, int(
            self.hS / 2))  # [4, B, 100/2] # number_of_layer_layer * (bi-direction) # lstm input convention.
        hidden = hidden.transpose(0, 1).contiguous()

        cell = self.W_cell(c_hs)  # [B, 4, 100/2]
        cell = cell.view(bS, self.lS * 2, int(self.hS / 2))  # [4, B, 100/2]
        cell = cell.transpose(0, 1).contiguous()

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=(hidden, cell),
                        last_only=False)  # [b, n, dim]

        att_n = self.W_att_n(wenc_n).squeeze(2)  # [B, max_len, 100] -> [B, max_len, 1] -> [B, max_len]

        #    Penalty
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_n[b, l_n1:] = -10000000000
        p_n = self.softmax_dim1(att_n)

        if show_p_wn:
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,6)
            cla()
            plot(p_n[0].data.numpy(), '--rs', ms=7)
            title('wn: nlu_weight')
            grid(True)
            fig.canvas.draw()

            show()
            # input('Type Enter to continue.')

        #    [B, mL_n, 100] *([B, mL_n] -> [B, mL_n, 1] -> [B, mL_n, 100] ) -> [B, 100]
        c_n = torch.mul(wenc_n, p_n.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_wn = self.wn_out(c_n)

        return s_wn

class WCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(WCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.mL_w = 20 # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_out = nn.Sequential(
            nn.Tanh(), nn.Linear(2 * hS, 20)
        )
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, show_p_wc, penalty=True):

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        att = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))

        mL_n = max(l_n)
        for b_n, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b_n, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)

        wenc_n = wenc_n.unsqueeze(1)  # [ b, n, dim] -> [b, 1, n, dim]

        p = p.unsqueeze(3)  # [b, hs, n] -> [b, hs, n, 1]
        c_n = torch.mul(wenc_n, p).sum(2)  # -> [b, hs, dim], c_n for each header.

        y = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)  # [b, hs, 2*dim]

        score = (self.W_out(y).squeeze(2)).transpose(1,2) # [b, hs]

        if penalty:
            for b, l_hs1 in enumerate(l_hs):
                score[b, :, l_hs1:] = -1e+10

        return score


class WOP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=7):
        super(WOP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 20 # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.wo_out = nn.Sequential(
            nn.Linear(2*hS, hS),
            nn.Tanh(),
            nn.Linear(hS, n_cond_ops)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wenc_n=None, show_p_wo=False):
        # Encode
        if not wenc_n:
            wenc_n = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=False,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        # wn

        wenc_hs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]

            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]] # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob) # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                              wenc_hs_ob.unsqueeze(3)
                              ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )

        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 5-1, dim] -> [bS, 5-1, 3]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_wo = self.wo_out(vec)
        return s_wo

class WAO(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_comb_ops=7):
        super(WAO, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 20 # max and/or number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.ao_out = nn.Sequential(
            nn.Linear(2*hS, hS),
            nn.Tanh(),
            nn.Linear(hS, n_comb_ops)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wenc_n=None, show_p_ao=False):
        # Encode
        if not wenc_n:
            wenc_n = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=False,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        # wn

        wenc_hs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]] # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob) # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                              wenc_hs_ob.unsqueeze(3)
                              ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )
        if show_p_ao:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wo: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 5-1, dim] -> [bS, 5-1, 3]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_ao = self.ao_out(vec)

        return s_ao

class WORD(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_order=5):  #n_order max_number of priority
        super(WORD, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 20 # max and/or number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.ord_out = nn.Sequential(
            nn.Linear(2*hS, hS),
            nn.Tanh(),
            nn.Linear(hS, n_order)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wenc_n=None, show_p_ord=False):
        # Encode
        if not wenc_n:
            wenc_n = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=False,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        # wn

        wenc_hs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]] # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob) # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                              wenc_hs_ob.unsqueeze(3)
                              ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )
        if show_p_ord:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wo: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 5-1, dim] -> [bS, 5-1, 3]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_ord = self.ord_out(vec)

        return s_ord

class WVP_se(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=7, old=False):
        super(WVP_se, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops

        self.mL_w = 20  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_op = nn.Linear(n_cond_ops, hS)

        if old:
            self.wv_out =  nn.Sequential(
            nn.Linear(4 * hS, 2)
            )
        else:
            self.wv_out = nn.Sequential(
                nn.Linear(4 * hS, hS),
                nn.Tanh(),
                nn.Linear(hS, 2)
            )
        # self.wv_out = nn.Sequential(
        #     nn.Linear(3 * hS, hS),
        #     nn.Tanh(),
        #     nn.Linear(hS, self.gdkL)
        # )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wenc_n=None, show_p_wv=False):

        # Encode
        if not wenc_n:
            wenc_n, hout, cout = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=True,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)

        wenc_hs_ob = []  # observed hs

        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)


        # Column attention
        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)
        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )

        if show_p_wv:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wv: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()


        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # Select observed headers only.
        # Also generate one_hot vector encoding info of the operator
        # [B, 4, dim]
        wenc_op = []
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0]) # not used anyway

            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)

            wenc_op.append(wenc_op1)

        # list to [B, 4, dim] tensor.
        wenc_op = torch.stack(wenc_op)  # list to tensor.
        wenc_op = wenc_op.to(device)

        # Now after concat, calculate logits for each token
        # [bS, 5-1, 3*hS] = [bS, 4, 300]
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)

        # Make extended vector based on encoded nl token containing column and operator information.

        vec1e = vec.unsqueeze(2).expand(-1,-1, mL_n, -1) # [bS, 4, 1, 300]  -> [bS, 4, mL, 300]
        wenc_ne = wenc_n.unsqueeze(1).expand(-1, 20, -1, -1) # [bS, 1, mL, 100] -> [bS, 4, mL, 100]
        vec2 = torch.cat( [vec1e, wenc_ne], dim=3)

        # now make logits
        s_wv = self.wv_out(vec2) # [bS, 4, mL, 400] -> [bS, 4, mL, 2]

        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                s_wv[b, :, l_n1:, :] = -10000000000
        return s_wv



class WBOOL(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(WBOOL, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 20 # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.wo_out = nn.Sequential(
            nn.Linear(2*hS, hS),
            nn.Tanh(),
            nn.Linear(hS, 2) # bool 0 or 1
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wenc_n=None, show_p_wb=False):
        # Encode
        if not wenc_n:
            wenc_n = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=False,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        # wn

        wenc_hs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]

            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]] # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob) # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                              wenc_hs_ob.unsqueeze(3)
                              ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )

        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 5-1, dim] -> [bS, 5-1, 3]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_wb = self.wo_out(vec)
        return s_wb




def Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_ao, s_ord, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, g_ao, g_ord, hs_type):
    """

    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    loss = 0
    loss += Loss_sc(s_sc, g_sc)
    loss += Loss_sa(s_sa, g_sa)
    loss += Loss_wn(s_wn, g_wn)
    loss += Loss_wc(s_wc, g_wn, g_wc)
    loss += Loss_wo(s_wo, g_wn, g_wo)
    loss += Loss_ao(s_ao, g_wn, g_ao)
    loss += Loss_ord(s_ord, g_wn, g_ord)
    loss += Loss_wv_se(s_wv, g_wn, g_wvi, g_wc, hs_type) #:temporary remove so we can ignore where value

    return loss

def Loss_sc(s_sc, g_sc):
    loss = F.cross_entropy(s_sc, torch.tensor(g_sc).to(device))
    return loss


def Loss_sa(s_sa, g_sa):
    loss = F.cross_entropy(s_sa, torch.tensor(g_sa).to(device))
    return loss

def Loss_wn(s_wn, g_wn):
    loss = F.cross_entropy(s_wn, torch.tensor(g_wn).to(device))

    return loss

def Loss_wc(s_wc, g_wn, g_wc):

    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wc1 = g_wc[b]
        s_wc1 = s_wc[b]
        loss += F.cross_entropy(s_wc1[:g_wn1], torch.tensor(g_wc1).to(device))

    return loss



def Loss_wo(s_wo, g_wn, g_wo):
    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1).to(device))

    return loss

def Loss_ao(s_ao, g_wn, g_ao):

    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_ao1 = g_ao[b]
        s_ao1 = s_ao[b]
        loss += F.cross_entropy(s_ao1[:(g_wn1)], torch.tensor(g_ao1).to(device))
    return loss

def Loss_ord(s_ord, g_wn, g_ord):
    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_ord1 = g_ord[b]
        s_ord1 = s_ord[b]
        loss += F.cross_entropy(s_ord1[:(g_wn1)], torch.tensor(g_ord1).to(device))
    return loss

def Loss_wv_se(s_wv, g_wn, g_wvi, g_wc, hs_type):
    """
    s_wv:   [bS, 4, mL, 2], 4 stands for maximum # of condition, 2 tands for start & end logits.
    g_wvi:  [ [1, 3, 2], [4,3] ] (when B=2, wn(b=1) = 3, wn(b=2) = 2).
    """
    loss = 0

    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        if g_wn1 == 0:
            continue

        g_wvi1 = torch.tensor(g_wvi1).to(device)

        g_st1 = g_wvi1[:,0]
        g_ed1 = g_wvi1[:,1]

        _,_,dim,_=s_wv.shape

        cnt=0
        for i in range(0,g_wn1):
            st=g_st1[i]
            ed=g_ed1[i]
            if (st == 1 and ed == 1) or (st == 0 and ed == 0):
                if st==1:
                    gt_ind=torch.tensor([[0.0,1.0]]).to(device)
                else:
                    gt_ind=torch.tensor([[1.0,0.0]]).to(device)

                predict = F.softmax(s_wv[b,cnt:cnt+1,dim-1,:],dim=1)

                loss = (predict - gt_ind).pow(2).sum()

                cnt=cnt+1
            else:

                loss += F.cross_entropy(s_wv[b,cnt:cnt+1,0:dim-1,0], g_st1[i:i+1])
                loss += F.cross_entropy(s_wv[b,cnt:cnt+1,0:dim-1,1], g_ed1[i:i+1])

                cnt=cnt+1

    return loss
