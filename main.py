import os
import pandas
import pandas as pd
import random
import numpy as np
import pydicom

import util
import shutil

# 1. datasource 로 df_pre_split 제작
# 2. processed 를 df_pre_split를 통해서 dataset 제작
# 필수조건 - dataset은 train val test 모두 각각 같은 환자들이 들어가게 fiesta와 ssfe로 나누어야한다.

log = util.LogMode('./log')
log.mode_print(True)


# pre split
# 1. datasource를 읽어서 case 수 수집(초기 df 제작), 이후 비율에 따라 train, validation, test 로 case를 나눈다.


def make_df_pre_split(path_src_='empty', path_dst_='empty', path_df_='empty'):
    log.add_line('RUN make_df_pre_split')
    if path_src_ == 'empty':
        log.add_line('ERROR ::: path is empty.')
        return 0
    lst_case = os.listdir(path_src_)

    lst_result = []
    for case in lst_case:
        lst_dcm = util.do_dcm(os.path.join(path_src_, case), path_dst_)
        util.do_binary(os.path.join(path_src_, case), path_dst_)

        lst_result += lst_dcm

    # lst_case = 데이터 들기 (FS 정보 들어가야함, )
    df = pd.DataFrame(lst_result)
    df.columns = ['case', 'mri type', 'fs', 'dcm file name']
    result = df.set_index('case')

    log.add_line('DataFrame path is... ' + path_df_)
    result.to_csv('df_result.csv')

    return result


def do_split(df_pre_split_, ratio_train=8, ratio_val=1, ratio_test=1):
    cases = df_pre_split_['case'].drop_duplicates()
    print(cases)
    n_total = len(cases)

    # split by ratio
    n_val = int((n_total * ratio_val / (ratio_train + ratio_val + ratio_test)) // 1)
    n_test = int((n_total * ratio_test / (ratio_train + ratio_val + ratio_test)) // 1)
    n_train = int(n_total - n_val - n_test)

    print(n_train, n_val, n_test)



path_src = os.path.join('./test_datas/datasource')
path_dst = os.path.join('./pre_split')

# df_pre_split = make_df_pre_split(path_src, path_dst, 'df_pre_split.csv')

print('load pre split')
df_pre_split = pd.read_csv('./df_result.csv')

# print(df_pre_split)

do_split(df_pre_split)