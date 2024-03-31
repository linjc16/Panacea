# Copyright 2019-present NAVER Corp.
# Apache License v2.0
# Wonseok modified. 20180607

# Xiaojing modified, 20190501

import os, sys, argparse, re, json
import numpy as np
from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

torch.cuda.set_device(4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=300, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int, # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers", #why 2... normally we only use last layer
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)

    #args.toy_model = not torch.cuda.is_available()
    args.toy_model = False #will change back
    args.toy_size = 12

    return args


def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):


    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')



    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    # if no_pretraining:
    #     pass
    # else:
        # model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        # print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config

def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert

def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', '>=', '<=', '<>', 'OP','Just']  # do not know why 'OP' required. Hence,
    comb_ops = ['and','or','+', '>', '>=', 'NULL','Just']
    comb_priority = ['0','1', '2', '3', '4','-1','Just']
    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)

    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    n_comb_ops=len (comb_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops, n_comb_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config

def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size, no_w2i=True, no_hs_tok=True)

    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train',epoch=None):
    model.train()
    model_bert.train()

    ave_loss = 0
    cnt = 0 # count the # of examples
    cnt_sc = 0 # count the # of correct predictions of select column
    cnt_sa = 0 # of selectd aggregation
    cnt_wn = 0 # of where number
    cnt_wc = 0 # of where column
    cnt_wo = 0 # of where operator
    cnt_wv = 0 # of where-value
    cnt_wvi = 0 # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0   # of execution acc
    cnt_ao = 0 # of and/or/>/>=/+ operrator acc
    cnt_ord = 0 # of order acc
    # Engine for SQL querying.
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(train_loader):
        cnt += len(t)

        if cnt < st_pos:
            continue

        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds, hs_type = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_ao, g_ord = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            #
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_ao, s_ord = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi, g_ao=g_ao, g_ord=g_ord, hs_type=hs_type )

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_ao, s_ord, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,g_ao,g_ord,hs_type)

        # Calculate gradient
        if iB % accumulate_gradients == 0: # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients-1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_ao, pr_ord = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_ao, s_ord,hs_type)

        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu, hs_type)

        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str,pr_ao,pr_ord, nlu,hs_type)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list,cnt_ao1_list, cnt_ord1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,g_ao,g_ord,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_ao, pr_ord,
                                                                   sql_i, pr_sql_i,hs_type,
                                                                   mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list,cnt_ao1_list, cnt_ord1_list,hs_type)
        # lx stands for logical form accuracy
        # Execution accuracy test.

        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i,pr_ao,pr_ord,hds)

        # statistics
        ave_loss += loss.item()

        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_ao += sum(cnt_ao1_list)
        cnt_ord += sum(cnt_ord1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        if cnt%100==0:
            print(cnt)
            print('avg loss:'+str(ave_loss/cnt)+'   ')
            print('epoch:'+str(epoch)+'\n')

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wv / cnt
    acc_wv = cnt_wv / cnt
    acc_ao = cnt_ao / cnt
    acc_ord = cnt_ord / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_ao, acc_ord, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out

def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_ao, g_ord, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_ao, pr_ord, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_ao, cnt_ord, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'pr_ao: {pr_ao}')
    print(f'g_ao : {g_ao}')
    print(f'pr_ord: {pr_ord}')
    print(f'g_ord : {g_ord}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(f'acc_lx = {cnt_lx/cnt:.3f}, acc_x = {cnt_x/cnt:.3f}\n',
          f'acc_sc = {cnt_sc/cnt:.3f}, acc_sa = {cnt_sa/cnt:.3f}, acc_wn = {cnt_wn/cnt:.3f}\n',
          f'acc_wc = {cnt_wc/cnt:.3f}, acc_wo = {cnt_wo/cnt:.3f}, acc_wv = {cnt_wv/cnt:.3f}\n',
          f'acc_ao = {cnt_ao/cnt:.3f}, acc_ord = {cnt_ord/cnt:.3f}')
    print(f'===============================')

def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1,
         path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_ao = 0
    cnt_ord = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    many=0
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds,hs_type = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_ao, g_ord = get_g(sql_i)

        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu,hs_type)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_ao, s_ord = model(wemb_n, l_n, wemb_h, l_hpu, l_hs, hs_type=hs_type)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_ao, s_ord, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,g_ao,g_ord,hs_type)


        # prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_ao, pr_ord = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, s_ao, s_ord,hs_type)
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu,hs_type)

        # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str,pr_ao,pr_ord, nlu,hs_type)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)


        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list,cnt_ao1_list, cnt_ord1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,g_ao,g_ord,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_ao, pr_ord,
                                                                   sql_i, pr_sql_i,hs_type,
                                                                   mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list,cnt_ao1_list, cnt_ord1_list,hs_type)


        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i,pr_ao,pr_ord,hds) # xiajing:need change

        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_ao += sum(cnt_ao1_list)
        cnt_ord += sum(cnt_ord1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_ao, cnt_ord, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_ao1_list, cnt_ord1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_ao = cnt_ao / cnt
    acc_ord = cnt_ord / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_ao, acc_ord, acc_lx, acc_x]
    return acc, results, cnt_list


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_ao, acc_ord, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f},acc_ao: {acc_ao:.3f},acc_ord: {acc_ord:.3f}, \
        acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}" )

if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = '/home'
    path_wikisql = os.path.join(path_h, 'Criteria2SQL', 'data/')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. Load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
    test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    test_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=test_data,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    ## 4. Build & Load models
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)

    ## 4.1.
    # To start from the pre-trained models, un-comment following lines.
    # path_model_bert =
    # path_model =
    # model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

    ## 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert, args.fine_tune)

    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    acc_result={}
    acc_result['epoch']=[]
    acc_result['train']=[]
    acc_result['dev']=[]
    acc_result['test']=[]
    with open('result.jsonl', 'w') as outfile:
        outfile.write('\n')
    with torch.no_grad():
        acc_dev, results_dev, cnt_list = test(dev_loader,
                                            dev_table,
                                            model,
                                            model_bert,
                                            bert_config,
                                            tokenizer,
                                            args.max_seq_length,
                                            args.num_target_layers,
                                            detail=False,
                                            path_db=path_wikisql,
                                            st_pos=0,
                                            dset_name='dev')
    for epoch in range(args.tepoch):
        # train
        acc_train, aux_out_train = train(train_loader,
                                         train_table,
                                         model,
                                         model_bert,
                                         opt,
                                         bert_config,
                                         tokenizer,
                                         args.max_seq_length,
                                         args.num_target_layers,
                                         args.accumulate_gradients,
                                         opt_bert=opt_bert,
                                         st_pos=0,
                                         path_db=path_wikisql,
                                         dset_name='train',
                                         epoch=epoch)

        # check DEV
        with torch.no_grad():
            acc_dev, results_dev, cnt_list = test(dev_loader,
                                                dev_table,
                                                model,
                                                model_bert,
                                                bert_config,
                                                tokenizer,
                                                args.max_seq_length,
                                                args.num_target_layers,
                                                detail=False,
                                                path_db=path_wikisql,
                                                st_pos=0,
                                                dset_name='dev')

            acc_test, results_test, cnt_list = test(test_loader,
                                                test_table,
                                                model,
                                                model_bert,
                                                bert_config,
                                                tokenizer,
                                                args.max_seq_length,
                                                args.num_target_layers,
                                                detail=False,
                                                path_db=path_wikisql,
                                                st_pos=0,
                                                dset_name='test')


        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'dev')
        print_result(epoch, acc_test, 'test')
        # save results for the official evaluation

        acc_result['epoch'].append(epoch)
        acc_result['train'].append(acc_train)
        acc_result['dev'].append(acc_dev)
        acc_result['test'].append(acc_test)
        with open('result.jsonl', 'a') as outfile:
            json.dump(acc_result, outfile)
            outfile.write('\n')

        # save results for the official evaluation
        save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')



        # save best model
        # Based on Dev Set logical accuracy lx
        acc_lx_t = acc_dev[-2]
        if acc_lx_t > acc_lx_t_best:
            acc_lx_t_best = acc_lx_t
            epoch_best = epoch
            # save best model
            state = {'model': model.state_dict()}
            torch.save(state, os.path.join('.', 'model_best.pt') )

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join('.', 'model_bert_best.pt'))

        print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
# python3.6 train.py --seed 1 --bS 2 --accumulate_gradients 2 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 512
