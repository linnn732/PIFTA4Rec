import torch
import numpy as np
import math
import random
from collections import defaultdict

# random.seed(11)

def get_dict(path):
    f = open(path, 'r')
    a = f.read()
    geted_dict = eval(a)
    f.close()
    return geted_dict

# @profile
def get_dataset(input_dir, max_basket_size,max_basket_num,neg_ratio,history = 0,next_k = 1): # 取得訓練集、驗證集、測試集
    print("--------------Begin Data Process--------------")
    neg_ratio = 1
    user_tran_date_dict_old = get_dict(input_dir)

    user_tran_date_dict = dict()
    for userid in user_tran_date_dict_old.keys():
        seq = user_tran_date_dict_old[userid]
        if len(seq) > max_basket_num: # 如果用戶的購物籃數超過駔大購物籃數，擷取後面的最大購物籃數量個購物籃
            seq = seq[-max_basket_num:]
        if len(seq) < 1 + next_k: continue # 如果購物籃序列少於2個，就忽略此用戶
        for b_id, basket in enumerate(seq): # 對每個購物籃擷取最大項目數
            if len(basket) > max_basket_size:
                seq[b_id] = basket[-max_basket_size:]
        user_tran_date_dict[userid] = seq   # 將購物籃序列存在對應此用戶的字典中


    train_times = 0
    valid_times = 0
    test_times = 0

    itemnum = 0
    for userid in user_tran_date_dict.keys():
        seq = user_tran_date_dict[userid]
        for basket in seq:
            for item in basket:
                if item > itemnum:
                    itemnum = item
    itemnum = itemnum + 1 # 最大項目數+1 = 項目總數
    item_list = [i for i in range(0, itemnum)] # 項目串列

    result_vector = np.zeros(itemnum)
    basket_count = 0
    for userid in user_tran_date_dict.keys():
        seq = user_tran_date_dict[userid][:-next_k] # 把每個用戶的最後一個購物籃之前的所有購物籃存到seq
        for basket in seq:
            basket_count += 1  # 計算購物籃總數
            result_vector[basket] += 1  # 每個項目出現的次數 (index是項目id，values是次數)
    weights = np.zeros(itemnum)
    max_freq = basket_count
    for idx in range(len(result_vector)):   # weights = 購物籃總數/項目次數
        if result_vector[idx] > 0:
            weights[idx] = max_freq / result_vector[idx]
        else:
            weights[idx] = 0

    TRAIN_DATASET = []
    train_batch = defaultdict(list)
    VALID_DATASET = []
    valid_batch = defaultdict(list)
    TEST_DATASET = []
    test_batch = defaultdict(list)
    neg_sample = dict()

    all_user_num = len(list(user_tran_date_dict.keys()))
    train_user_num = 0
    train_userid_list = list(user_tran_date_dict.keys())[:math.ceil(0.9 * len(list(user_tran_date_dict.keys())))] # 取出90%的用戶作為訓練集

    # 產生每個用戶的負樣本
    for userid in user_tran_date_dict.keys():
        seq = user_tran_date_dict[userid][:-next_k]  # 取最後一個購物籃之前的所有購物籃
        seq_pool = []
        for basket in seq:
            seq_pool = seq_pool + basket
        neg_sample[userid] = list(set(item_list) - set(seq_pool))  # 每個用戶的負樣本集合


    for userid in user_tran_date_dict.keys():
        if userid in train_userid_list:
            seq = user_tran_date_dict[userid]
            before = []
            train_seq = seq[:-1]
            for basketid, basket in enumerate(train_seq):
                if len(basket) > max_basket_size:
                    basket = basket[-max_basket_size:]
                else: # 加上padding讓每個用戶的每個購物籃大小相同。
                    padd_num = max_basket_size - len(basket)
                    padding_item = [-1] * padd_num
                    basket = basket + padding_item
                before.append(basket)
                if len(before) == 1: continue
                U = userid  
                S = before[:-1]  # 將此用戶目前的購物籃作為標籤，前面的所有購物籃進行訓練
                S_pool = []
                H = np.zeros(itemnum)
                H_pad = np.zeros(itemnum + 1)
                for basket in S:    # 這邊的basket是目前購物籃之前的所有購物籃
                    S_pool = S_pool + basket
                    no_pad_basket = list(set(basket)-set([-1]))
                    H[no_pad_basket] += 1   # 此用戶購買歷史中，每個項目的出現次數
                H = H / len(before[:-1])    # 計算每个物品的平均出现次数
                H_pad[1:] = H
                L = len(before[:-1])        # 此用戶歷史購物籃長度
                tar_basket = train_seq[basketid]    # 目前的購物籃作為標籤
                # print("H_pad=",H_pad)
                for item in tar_basket:
                    T = item  
                    N = random.sample(neg_sample[userid], neg_ratio)    # 負樣本
                    train_batch[L].append((U, S_pool, T, H_pad[0:2], N, L))     # train_batch =>  user_id、購買歷史項目、目標購物籃的一個項目、H_pad、負樣本、購物籃長度
                    train_times += 1

            test_seq = seq
            before = []
            for basketid, basket in enumerate(test_seq):
                if len(basket) > max_basket_size:
                    basket = basket[-max_basket_size:]
                else:
                    padd_num = max_basket_size - len(basket)
                    padding_item = [-1] * padd_num
                    basket = basket + padding_item
                before.append(basket)
            U = userid
            S = list(before[:-1])
            S_pool = []
            H = np.zeros(itemnum)
            H_pad = np.zeros(itemnum+1)
            for basket in S:
                S_pool = S_pool + basket
                no_pad_basket = list(set(basket) - set([-1]))
                H[no_pad_basket] += 1
            H = H / len(S)
            H_pad[1:] = H
            L = len(before[:-1])
            T_basket = before[-1]
            test_batch[L].append((U, S_pool, T_basket, H_pad[0:2], L))
            test_times += 1
          

        else:
            seq = user_tran_date_dict[userid]
            before = []
            valid_seq = seq
            for basketid, basket in enumerate(valid_seq): # valid_seq: (驗證集)此用戶的所有購物籃序列
                if len(basket) > max_basket_size:
                    basket = basket[-max_basket_size:]
                else:
                    padd_num = max_basket_size - len(basket)
                    padding_item = [-1] * padd_num
                    basket = basket + padding_item
                before.append(basket)
                if len(before) == 1: continue
                if len(before) < len(valid_seq): continue # 將此用戶所有購物籃都裝到before中再往下執行
                U = userid  
                S = before[:-1] 
                S_pool = []
                H = np.zeros(itemnum)
                H_pad = np.zeros(itemnum + 1)
                for basket in S:
                    S_pool = S_pool + basket
                    no_pad_basket = list(set(basket) - set([-1]))
                    H[no_pad_basket] += 1
                H = H / len(S)
                H_pad[1:] = H
                L = len(before[:-1])  
                tar_basket = valid_seq[basketid]    # 此用戶最後一個購物籃是要預測的目標購物籃

                if history == 0:
                    tar_basket = list(set(tar_basket)-set(S_pool)) # 要預測沒買過的，所以移除有出現在歷史購物籃中的項目
                    if len(tar_basket) < 1:continue
                    padd_num = max_basket_size - len(tar_basket)
                    padding_item = [-1] * padd_num
                    T_basket = tar_basket + padding_item
                    valid_batch[L].append((U, S_pool, T_basket, H_pad[0:2], L))
                    valid_times += 1
                else:
                    T_basket = before[-1]
                    valid_batch[L].append((U, S_pool, T_basket, H_pad[0:2], L))
                    valid_times += 1

    for l in train_batch.keys():
        TRAIN_DATASET.append(list(zip(*train_batch[l])))

    for l in test_batch.keys():
        TEST_DATASET.append(list(zip(*test_batch[l])))

    for l in valid_batch.keys():
        VALID_DATASET.append(list(zip(*valid_batch[l])))



    print("--------------Data Process is Over--------------")
    return TRAIN_DATASET, VALID_DATASET, TEST_DATASET, neg_sample, weights, itemnum, train_times, test_times, valid_times



# @profile
def get_batch_TRAIN_DATASET(dataset, batch_size):
    print('--------------Data Process is Begin--------------')
    random.shuffle(dataset)
    for idx, (UU, SS, TT, HH, NN, LL) in enumerate(dataset):
        userid = torch.tensor(UU, dtype=torch.long)
        input_seq = torch.tensor(SS, dtype=torch.long)  
        target = torch.tensor(TT, dtype=torch.long) 
        history = torch.from_numpy(np.array(HH)).float() 
        neg_items = torch.tensor(NN, dtype=torch.long)  

        if SS.__len__() < 2:
            continue
        if SS.__len__() <= batch_size:
            batch_userid = userid
            batch_input_seq = input_seq
            batch_target = target
            batch_history = history
            batch_neg_items = neg_items
            yield (batch_userid,batch_input_seq,batch_target,batch_history,batch_neg_items)
        else:
            batch_begin = 0
            while (batch_begin + batch_size) <= SS.__len__():
                batch_userid = userid[batch_begin:batch_begin + batch_size]
                batch_input_seq = input_seq[batch_begin:batch_begin + batch_size]
                batch_target = target[batch_begin:batch_begin + batch_size]
                batch_history = history[batch_begin:batch_begin + batch_size]
                batch_neg_items = neg_items[batch_begin:batch_begin + batch_size]
                yield (batch_userid, batch_input_seq, batch_target, batch_history, batch_neg_items)
                batch_begin = batch_begin + batch_size
            if (batch_begin + batch_size > SS.__len__()) & (batch_begin < SS.__len__()):

                batch_userid = userid[batch_begin:]
                batch_input_seq = input_seq[batch_begin:]
                batch_target = target[batch_begin:]
                batch_history = history[batch_begin:]
                batch_neg_items = neg_items[batch_begin:]
                yield (batch_userid, batch_input_seq, batch_target, batch_history, batch_neg_items)




# @profile
def get_batch_TEST_DATASET(TEST_DATASET, batch_size):
    BATCHES = []
    random.shuffle(TEST_DATASET)
    for idx, (UU, SS, TT_bsk, HH, LL) in enumerate(TEST_DATASET): 
    
        userid = torch.tensor(UU, dtype=torch.long)
        input_seq = torch.tensor(SS, dtype=torch.long)
        try:
            target = torch.tensor(TT_bsk, dtype=torch.long)
        except ValueError:
            print(TT_bsk)
        history = torch.from_numpy(np.array(HH)).float() 

        assert UU.__len__() == SS.__len__()
        assert UU.__len__() == TT_bsk.__len__()
        assert UU.__len__() == HH.__len__()

        if SS.__len__() < 1: continue
        if SS.__len__() <= batch_size:
            batch_userid = userid
            batch_input_seq = input_seq
            batch_target = target
            batch_history = history
            yield (batch_userid, batch_input_seq, batch_target, batch_history)
        else:
            batch_begin = 0
            while (batch_begin + batch_size) <= SS.__len__():
                batch_userid = userid[batch_begin:batch_begin + batch_size]
                batch_input_seq = input_seq[batch_begin:batch_begin + batch_size]
                batch_target = target[batch_begin:batch_begin + batch_size]
                batch_history = history[batch_begin:batch_begin + batch_size]
                yield (batch_userid, batch_input_seq, batch_target, batch_history)
                batch_begin = batch_begin + batch_size

            if (batch_begin + batch_size > SS.__len__()) & (batch_begin < SS.__len__()):
                batch_userid = userid[batch_begin:]
                batch_input_seq = input_seq[batch_begin:]
                batch_target = target[batch_begin:]
                batch_history = history[batch_begin:]
                yield (batch_userid, batch_input_seq, batch_target, batch_history)