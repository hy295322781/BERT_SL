# -*- coding:utf-8 -*-
"""
Created on 2019-11-13 21:50:45
Author: Xiong Zecheng (295322781@qq.com)
"""
def extract_slot(seq):
    slot = list()
    i = 0
    while i < len(seq):
        if seq[i].startswith("B"):
            slot_type = seq[i][2:]
            l = 1
            while i + l < len(seq) and seq[i + l] == "I-" + slot_type:
                l += 1
            slot.append((i, slot_type, l))  # (槽的位置,槽的类型,槽的长度)
            i = i + l
        else:
            i += 1
    return slot

def convert_slot(tuple_slot,tokens):
    '''
    将三元组的槽转换为键值对槽
    '''
    pair_slot = list()
    for i,slot_type,l in tuple_slot:
        slot_value = "".join(tokens[i:i+l])
        pair = (slot_type,slot_value)
        if pair not in pair_slot:
            pair_slot.append(pair)
    return pair_slot


def evaluate(result_file,origin_examples,error_example_output,true_example_output,distinct=False):
    '''
    :param distinct: 是否将一个样本中的重复槽视为一个
    '''
    with open(result_file,'r') as fr:
        lines = fr.readlines()
    slot_hit_count = 0
    predict_slot_count = 0
    actual_slot_count = 0
    efw = open(error_example_output,'w')
    tfw = open(true_example_output,'w')
    for i,predict_line in enumerate(lines):
        is_true = True
        example = origin_examples[i]
        predict_seq = predict_line.strip().split(" ")
        actual_seq = example.label.strip().split(" ")
        predict_slot = extract_slot(predict_seq)
        actual_slot = extract_slot(actual_seq)
        if distinct:
            text_seq = example.text.strip().split(" ")
            predict_slot = convert_slot(predict_slot,text_seq)
            actual_slot = convert_slot(actual_slot, text_seq)
        for item in predict_slot:
            if item in actual_slot:
                slot_hit_count += 1
            else:
                is_true = False
        if not is_true:
            efw.write("{} actual_slot:{} predict_slot:{}\n".format(i+1,actual_slot,predict_slot))
            efw.write(example.text+"\n")
            efw.write(example.label + "\n")
            efw.write(predict_line + "\n")
        else:
            tfw.write("{} {} {}\n".format(i+1,example.text.replace(" ",""),actual_slot))
        predict_slot_count += len(predict_slot)
        actual_slot_count += len(actual_slot)
    efw.close()
    print("test set size:{}".format(len(origin_examples)))
    print("predicted slot:{} actual slot:{} hit:{}".format(predict_slot_count,actual_slot_count,slot_hit_count))
    print("Precision:{}".format(slot_hit_count/predict_slot_count))
    print("Recall:{}".format(slot_hit_count/actual_slot_count))
    print("F1score:{}".format(2 * slot_hit_count / (actual_slot_count + predict_slot_count)))