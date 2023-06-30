#!/usr/bin/env python
# coding: utf-8
# usage: require 1. ref_human_detail 2. human_our_detail 3 ref_our_detail

import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


out_put_f = open('mdd_result', 'w')
f = open("ref_human_detail",'r')
dic={}
insert = 0 
delete = 0
sub = 0
cor=0
count=0
##  0： ref  1：human 2：ops --- 3: human  4： our  5: ops 
for line in f:
    line = line.strip()
    if("ref" in line ):
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +"," ",ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]] = []
        dic[ref[0]].append(ref[1])
    elif( "hyp" in line ):
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +"," ",hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif( " op " in line ):   
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +"," ",op[1])
        op_seq = op[1].split(" ")
        
        for i, o in enumerate(op_seq):
            if o == 'S':
                sub += 1
            elif o == 'D':
                delete += 1 
            elif o == 'I':
                insert += 1
            else:
                cor += 1
            count += 1
        dic[op[0]].append(" ".join(op_seq))

f.close()
## 发音错误统计
print("insert:" ,insert, file=out_put_f)
print("delete:" ,delete, file=out_put_f)
print("sub:" ,sub, file=out_put_f)
print("cor:" ,cor, file=out_put_f)
print("sum:", count, file=out_put_f)

S = 0
D = 0
I = 0
N = 0
C = 0
phone_our = []
phone_human = []
f = open("human_our_detail",'r')
for line in f:
    line = line.strip()
    fn = line.split(" ")[0]
    if(fn not in dic):
        continue
    if("ref" in line ):
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +"," ",ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]].append(ref[1])
    elif( "hyp" in line ):
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +"," ",hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif( " op " in line ):
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +"," ",op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])

        # performance of phone recognition
        S += len([x for x in op_seq if x == 'S'])
        D += len([x for x in op_seq if x == 'D'])
        I += len([x for x in op_seq if x == 'I'])
        C += len([x for x in op_seq if x == 'C'])

        for i in range(len(ref_seq)):
            if ref_seq[i] in ('<eps>', 'UNK') or hyp_seq[i] in ('<eps>', 'UNK'):
                continue
            phone_human.append(ref_seq[i])
            phone_our.append(hyp_seq[i])
        
f.close()
N = C + S + D
print(f'N: {N}', file=out_put_f)
print(f'S: {S}', file=out_put_f)
print(f'D: {D}', file=out_put_f)
print(f'I: {I}', file=out_put_f)
print(f'Correct.: {(N-S-D)/N:.2%}', file=out_put_f)
print(f'Acc.: {(N-S-D-I)/N:.2%}', file=out_put_f)
class_list = list(set(phone_human) | set(phone_our))
class_list.sort()
cm = confusion_matrix(phone_human, phone_our, labels=class_list, normalize="true")

df_cm = pd.DataFrame(cm, index = class_list, columns = class_list)
plt.figure(figsize = (20,16))
sns.heatmap(df_cm, annot=False).get_figure().savefig("confusion_matrix.png")
cm_origin = confusion_matrix(phone_human, phone_our, labels=class_list)
df_cm_origin = pd.DataFrame(cm_origin, index = class_list, columns = class_list)
df_cm_origin.to_csv('confusion_matrix.csv')

star_list = [x for x in class_list if '*' in x]
remove_star = str.maketrans('', '', '*')
star_correspondence_list = [x.translate(remove_star) for x in star_list]
mix_list = star_list + star_correspondence_list
mix_list.sort()
df_star_cm = df_cm.loc[mix_list, mix_list]
plt.figure(figsize = (20,16))
sns.heatmap(df_star_cm, annot=False).get_figure().savefig("star_confusion_matrix.png")

consonant_minimal_pair_list = ['d', 'dh', 't', 'sh', 's', 'z']
vowel_minimal_pair_list = ['aa', 'ah', 'ae', 'eh', 'ih', 'iy']
df_star_cm = df_cm.loc[consonant_minimal_pair_list, consonant_minimal_pair_list]
plt.figure(figsize = (20,16))
sns.heatmap(df_star_cm, annot=False).get_figure().savefig("consonant_confusion_matrix.png")
df_star_cm = df_cm.loc[vowel_minimal_pair_list, vowel_minimal_pair_list]
plt.figure(figsize = (20,16))
sns.heatmap(df_star_cm, annot=False).get_figure().savefig("vowel_confusion_matrix.png")

l = []
for i in range(len(class_list)):
    for j in range(len(class_list)):
        if i == j:
            continue
        l.append((class_list[i], class_list[j], df_cm_origin.values[i, j]))
df = pd.DataFrame(l, columns=('label', 'pred', 'proportion')).sort_values('proportion', ascending=False)
df.to_csv('flattened_confusion_matrix.csv', index=None)


f = open("ref_our_detail",'r')
for line in f:
    line = line.strip()
    fn = line.split(" ")[0]
    if(fn not in dic):
        continue
    if("ref" in line ):
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +"," ",ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]].append(ref[1])
    elif( "hyp" in line ):
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +"," ",hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif( " op " in line ):
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +"," ",op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])
f.close()


cor_cor = 0
# cor_nocor = 0
sub_sub = 0
sub_different_sub = 0
sub_nosub = 0
ins_ins = 0
ins_noins = 0
ins_different_ins = 0
del_del = 0
del_nodel = 0
mis_cor = 0
mis_diag_ins = 0
cor_del = 0
cor_sub = 0
test_sum = 0
ins_cor = 0
del_cor = 0
sub_cor = 0

detail_dict = dict()
for key, arr in dic.items():
    detail_list = list()

    ref_label_arr = np.array([x.split(' ') for x in arr[:3]])
    label_our_arr = np.array([x.split(' ') for x in arr[3:6]])
    ref_our_arr = np.array([x.split(' ') for x in arr[-3:]])

    # for i in range(ref_label_arr.shape[1]):
    empty_block = np.array([['',], ['',], ['',]])
    pos1, pos2 = 0, 0
    l1, l2 = [], []
    for pos in range(ref_our_arr[:, ref_our_arr[0, :]!='<eps>'].shape[1]):
        tmp1, tmp2 = [], []        
        while ref_label_arr[0, pos1] == '<eps>':
            tmp1.append(pos1)
            pos1 += 1
        tmp1.append(pos1)
        while ref_our_arr[0, pos2] == '<eps>':
            tmp2.append(pos2)
            pos2 += 1
        tmp2.append(pos2)
        if len(tmp1)-1 > 0 and len(tmp2)-1 > 0:
            if len(tmp1) == len(tmp2):
                l1.append(ref_label_arr[:, tmp1])
                l2.append(ref_our_arr[:, tmp2])
            elif len(tmp1) > len(tmp2):
                l1.append(ref_label_arr[:, tmp1])
                l2.append(np.concatenate([*[empty_block]*(len(tmp1)-len(tmp2)), ref_our_arr[:, tmp2]], axis=1))
            else:
                l1.append(np.concatenate([*[empty_block]*(len(tmp2)-len(tmp1)), ref_label_arr[:, tmp1]], axis=1))
                l2.append(ref_our_arr[:, tmp2])
        elif len(tmp1)-1 > 0:
            l1.append(ref_label_arr[:, tmp1])
            l2.append(np.concatenate([*[empty_block]*(len(tmp1)-len(tmp2)), ref_our_arr[:, tmp2]], axis=1))
        elif len(tmp2)-1 > 0:
            l1.append(np.concatenate([*[empty_block]*(len(tmp2)-len(tmp1)), ref_label_arr[:, tmp1]], axis=1))
            l2.append(ref_our_arr[:, tmp2])
        else:
            l1.append(ref_label_arr[:, tmp1])
            l2.append(ref_our_arr[:, tmp2])
        
        pos1 += 1
        pos2 += 1

    len_appendix_1 = ref_label_arr.shape[1] - pos1
    len_appendix_2 = ref_our_arr.shape[1] - pos2
    if len_appendix_1 > 0 and len_appendix_2 > 0:
        if len_appendix_1 == len_appendix_2:
            l1.append(ref_label_arr[:, pos1-1:])
            l2.append(ref_our_arr[:, pos2-1:])
        elif len_appendix_1 > len_appendix_2:
            l1.append(ref_label_arr[:, pos1-1:])
            l2.append(np.concatenate([ref_our_arr[:, pos2-1:], *[empty_block]*(len_appendix_1 - len_appendix_2)], axis=1))
        else:
            l1.append(np.concatenate([ref_label_arr[:, pos1-1:], *[empty_block]*(len_appendix_2 - len_appendix_1)], axis=1))
            l2.append(ref_our_arr[:, pos2-1:])
    elif len_appendix_1 > 0:
        l1.append(ref_label_arr[:, pos1:])
        l2.append(np.concatenate([*[empty_block]*len_appendix_1], axis=1))
    elif len_appendix_2 > 0:
        l1.append(np.concatenate([*[empty_block]*len_appendix_2], axis=1))
        l2.append(ref_our_arr[:, pos2:])
    
    aligned_arr = np.concatenate([np.concatenate(l1, axis=1), np.concatenate(l2, axis=1)], axis=0) 

    for i in range(aligned_arr.shape[1]):
        if aligned_arr[2, i] == 'C' or aligned_arr[2, i] == '':
            continue
        d = dict()
        d['error'] = ','.join([aligned_arr[0, i], aligned_arr[1, i], aligned_arr[2, i]])
        d['correct'] = aligned_arr[4, i] == aligned_arr[1, i] and aligned_arr[5, i] == aligned_arr[2, i]
        d['diagnosis'] = ','.join([aligned_arr[3, i], aligned_arr[4, i], aligned_arr[5, i]])
        detail_list.append(d)
    detail_dict[key] = detail_list

    del_del += aligned_arr[:, (aligned_arr[2, :]=='D')&(aligned_arr[5, :]=='D')].shape[1]
    del_nodel += aligned_arr[:, (aligned_arr[2, :]=='D')&(aligned_arr[5, :]!='D')&(aligned_arr[5, :]!='C')].shape[1]    
    
    ins_ins += aligned_arr[:, (aligned_arr[2, :]=='I')&(aligned_arr[5, :]=='I')&(aligned_arr[1, :]==aligned_arr[4, :])].shape[1]
    ins_noins += aligned_arr[:, (aligned_arr[2, :]=='I')&(aligned_arr[5, :]!='I')&(aligned_arr[5, :]!='C')].shape[1]
    mis_diag_ins += aligned_arr[:, (aligned_arr[2, :]!='I')&(aligned_arr[5, :]=='I')&(aligned_arr[4, :]!='sil')].shape[1]
    ins_different_ins += aligned_arr[:, (aligned_arr[2, :]=='I')&(aligned_arr[5, :]=='I')&(aligned_arr[1, :]!=aligned_arr[4, :])].shape[1]

    sub_sub += aligned_arr[:, (aligned_arr[2, :]=='S')&(aligned_arr[5, :]=='S')&(aligned_arr[1, :]==aligned_arr[4, :])].shape[1]
    sub_nosub += aligned_arr[:, (aligned_arr[2, :]=='S')&(aligned_arr[5, :]!='S')&(aligned_arr[5, :]!='C')].shape[1]
    sub_different_sub += aligned_arr[:, (aligned_arr[2, :]=='S')&(aligned_arr[5, :]=='S')&(aligned_arr[1, :]!=aligned_arr[4, :])].shape[1]

    cor_cor += aligned_arr[:, (aligned_arr[2, :]=='C')&(aligned_arr[5, :]=='C')].shape[1]
    # cor_nocor += aligned_arr[:, ((aligned_arr[2, :]=='C')&aligned_arr[5, :]=='I')].shape[1]
    cor_del += aligned_arr[:, (aligned_arr[2, :]=='C')&(aligned_arr[5, :]=='D')&(aligned_arr[3, :]!='sil')].shape[1]
    cor_sub += aligned_arr[:, (aligned_arr[2, :]=='C')&(aligned_arr[5, :]=='S')].shape[1]

    mis_cor += aligned_arr[:, (aligned_arr[2, :]!='C')&(aligned_arr[5, :]=='C')&(aligned_arr[1, :]!='sil')].shape[1]

    ins_cor += aligned_arr[:, (aligned_arr[2, :]=='I')&(aligned_arr[5, :]=='C')&(aligned_arr[1, :]!='sil')].shape[1]
    del_cor += aligned_arr[:, (aligned_arr[2, :]=='D')&(aligned_arr[5, :]=='C')&(aligned_arr[1, :]!='sil')].shape[1]
    sub_cor += aligned_arr[:, (aligned_arr[2, :]=='S')&(aligned_arr[5, :]=='C')&(aligned_arr[1, :]!='sil')].shape[1]

    test_sum += aligned_arr.shape[1]


true_acceptance = cor_cor
false_rejection = cor_del + cor_sub + mis_diag_ins
false_acceptance = mis_cor
correct_diagnosis = del_del + ins_ins + sub_sub
diagnosis_error = sub_nosub + ins_noins + del_nodel + sub_different_sub + ins_different_ins
true_rejection = correct_diagnosis + diagnosis_error
correct_pronunciations = true_acceptance + cor_del + cor_sub
mispronunciations = false_acceptance + true_rejection

FRR = false_rejection / correct_pronunciations
FAR = mis_cor / mispronunciations
DER = diagnosis_error / true_rejection
Detection_Rate = (true_rejection+true_acceptance) / (true_rejection + false_rejection + true_acceptance + false_acceptance)
Precision = true_rejection / (true_rejection + false_rejection)
Recall = 1 - FAR
F_measure = 2 * Precision * Recall / (Precision + Recall)

SUB_CDR = sub_sub / (sub_sub + sub_different_sub + sub_nosub)
INS_CDR = ins_ins / (ins_ins + ins_noins + mis_diag_ins + ins_different_ins)
DEL_CDR = del_del / (del_del + del_nodel + cor_del)

print("mis_diag_ins: ", mis_diag_ins, file=out_put_f)
print("cor_cor: ", cor_cor, file=out_put_f)
print("cor_nocor: ", cor_del + cor_sub, file=out_put_f)
print("sub_sub: ", sub_sub, file=out_put_f)
print("sub_nosub: ", sub_nosub, file=out_put_f)
print("sub_different_sub: ", sub_different_sub, file=out_put_f)
print("ins_different_ins: ", ins_different_ins, file=out_put_f)
print("ins_ins: ", ins_ins, file=out_put_f)
print("ins_noins: ", ins_noins, file=out_put_f)
print("del_del: ", del_del, file=out_put_f)
print("del_nodel: ", del_nodel, file=out_put_f)
print("ins_cor: ", ins_cor, file=out_put_f)
print("del_cor: ", del_cor, file=out_put_f)
print("sub_cor: ", sub_cor, file=out_put_f)
print("mis_cor: ", mis_cor, file=out_put_f)
print("test_sum:", test_sum, file=out_put_f)

print(f"DEL_CDR: {DEL_CDR:.2%}", file=out_put_f)
print(f"INS_CDR: {INS_CDR:.2%}", file=out_put_f)
print(f"SUB_CDR: {SUB_CDR:.2%}", file=out_put_f)

print(f"FRR: {FRR:.2%}", file=out_put_f)
print(f"FAR: {FAR:.2%}", file=out_put_f)
print(f"Detection_Rate: {Detection_Rate:.2%}", file=out_put_f)
print(f"DER: {DER:.2%}", file=out_put_f)
print(f"Recall: {Recall:.2%}", file=out_put_f)
print(f"Precision: {Precision:.2%}", file=out_put_f)
print(f"F-measure: {F_measure:.2%}", file=out_put_f)
out_put_f.close()

result_dict = {
    "Correct.": (N-S-D)/N,
    "Acc.": (N-S-D-I)/N,
    "mis_diag_ins": mis_diag_ins,
    "cor_cor": cor_cor,
    "cor_nocor": cor_del + cor_sub,
    "sub_sub": sub_sub,
    "sub_nosub": sub_nosub,
    "sub_different_sub": sub_different_sub,
    "ins_ins": ins_ins,
    "ins_noins": ins_noins,
    "ins_different_ins": ins_different_ins,
    "del_del": del_del,
    "del_nodel": del_nodel,
    "ins_cor": ins_cor,
    "del_cor": del_cor,
    "sub_cor": sub_cor,
    "mis_cor": mis_cor,
    "DEL_CDR": DEL_CDR,
    "INS_CDR": INS_CDR,
    "SUB_CDR": SUB_CDR,
    "FRR": FRR,
    "FAR": FAR,
    "DER": DER,
    "Detection Rate": Detection_Rate,
    "Recall": Recall,
    "Precision": Precision,
    "F-measure": F_measure,
}

with open("result_dict", "wb") as f:
    pickle.dump(result_dict, f)

with open("detail_dict", 'wb') as f:
    pickle.dump(detail_dict, f)