# preprocess.py
# 전처리 과정 모듈화 -> 수정과 재사용의 용이성을 위해 객체지향적 코드 구성


import os
import re  # regular Ex
import json  # to read json

import numpy as np  # numpy calculation
import pandas as pd  # data frame
from tqdm import tqdm  # progress bar

from konlpy.tag import Okt  # Korean Morphs


# 학습에 사용할 데이터를 위한 데이터 처리 관련 설정
FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"   # 패딩
STD = "<SOS>"   # 시작 토큰
END = "<END>"   # 종료 토큰
UNK = "<UNK>"   # OOV 토큰

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)  # 미리 컴파일하면 패턴을 사용할 때 반복적으로 컴파일 하는데 드는 시간 줄일 수 있음

MAX_SEQUENCE = 25


def load_data(path):
    """데이터 로딩 함수"""
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer


def data_tokenizer(data):
    """토큰화 함수"""
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence) # 특수 기호 모두 제거
        for word in sentence.split():  # 공백문자를 기준으로 단어 나눔
            words.append(word)

    return [word for word in words if word]  # 전체 데이터의 모든 단어를 포함하여 단어 리스트로 만듦


def prepro_like_morphlized(data):
    """형태소 분리기를 통한 데이터 토큰화 함수"""
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data


def make_vocabulary(vocab_list):
    # 리스트를 키가 단어이고 값이 인덱스인 딕셔너리로 변환
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}

    # 키가 인덱스, 값이 단어인 딕셔너리
    idx2word = {idx: word for idx, word in enumerate(vocab_list)}

    # 두개 리스트 반환
    return word2idx, idx2word


def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    """단어 사전 생성을 위한 함수"""
    vocab_list = []

    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            if tokenize_as_morph:
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)

            data = []
            data.extend(question)
            data.extend(answer)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER

        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocab_list.append(line.strip())

    word2idx, idx2word = make_vocabulary(vocab_list)

    return word2idx, idx2word, len(word2idx)


def enc_processing(value, dictionary, tokenize_as_morph=False):
    """인코더 부분에 사용될 전처리 함수"""
    sequences_input_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])

        if len(sequence_index) > MAX_SEQUENCE:
            sequence = sequence_index[:MAX_SEQUENCE]

        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]

        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index), sequences_length


def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    """디코더의 입력값을 만드는 함수
       ex) '안녕 오랜만이야' 라는 문장에 대해서
          디코더 입력값: '<SOS>, 그래, 오랜만이야.<PAD>'  <<<<<<<<<< 이 처리를 위한 함수
          디코더 타겟값: '그래, 오랜만이야. <END>, <PAD>'

       입력값으로 시작 토큰이 앞에 들어감
    """
    sequences_output_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]

        sequences_output_index.append(sequence_index)
    return np.asarray(sequences_output_index), sequences_length


# def dec_target_processing(value, dictionary, tokenize_as_morph=False):
#     """디코더의 입력값을 만드는 함수
#        ex) '안녕 오랜만이야' 라는 문장에 대해서
#           디코더 입력값: '<SOS>, 그래, 오랜만이야.<PAD>'
#           디코더 타겟값: '그래, 오랜만이야. <END>, <PAD>'  <<<<<<<<<< 이 처리를 위한 함수
#
#        디코더 입력값을 만드는 함수와의 차이점은 문장이 시작하는 부분에 토큰을 넣지 않고 마지막에 종료 토큰을 넣는다는 점.
#        """
#     sequences_target_index = []
#     if tokenize_as_morph:
#         value = prepro_like_morphlized(value)
#
#     for sequence in value:
#         sequence = re.sub(CHANGE_FILTER, "", sequence)
#         sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
#         if len(sequence_index) >= MAX_SEQUENCE:
#             sequence_index = sequence_index[:MAX_SEQUENCE-1] + [dictionary[END]]
#         else:
#             sequence_index += dictionary[END]
#
#         sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]] # 인덱스보다 짧은 경우
#         sequences_target_index.append(sequence_index)
#
#     return np.asarray(sequences_target_index)

def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    """디코더의 입력값을 만드는 함수
    ex) '안녕 오랜만이야' 라는 문장에 대해서
          디코더 입력값: '<SOS>, 그래, 오랜만이야.<PAD>'
         디코더 타겟값: '그래, 오랜만이야. <END>, <PAD>'  <<<<<<<<<< 이 처리를 위한 함수

         디코더 입력값을 만드는 함수와의 차이점은 문장이 시작하는 부분에 토큰을 넣지 않고 마지막에 종료 토큰을 넣는다는 점.
    """
    sequences_target_index = []
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE-1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)