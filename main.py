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
    result = df.reset_index(drop=True)

    log.add_line('DataFrame path is... ' + path_df_)
    result.to_csv(path_df_)

    log.add_line('Done make_df_pre_split\n')
    return result


def do_split(df_pre_split_, path_splitted, ratio_train=8, ratio_val=1, ratio_test=1):
    log.add_line('RUN do_split')
    print()
    cases = df_pre_split_['case'].drop_duplicates().reset_index(drop=True)

    print(cases)

    n_total = len(cases)
    log.add_line('Total : %d' % n_total)

    # split by ratio
    n_val = int((n_total * ratio_val / (ratio_train + ratio_val + ratio_test)) // 1)
    n_test = int((n_total * ratio_test / (ratio_train + ratio_val + ratio_test)) // 1)
    n_train = int(n_total - n_val - n_test)

    log.add_line('train : %d, validation : %d, test %d' % (n_train, n_val, n_test))

    df_shuffle = pd.DataFrame({'case': cases.sample(frac=1).reset_index(drop=True)})

    df_learn_type = pd.DataFrame({'learn type': ['train'] * n_train + ['val'] * n_val + ['test'] * n_test})
    df_shuffle.insert(0, 'learn type', df_learn_type)

    df_splitted = pd.merge(df_pre_split_, df_shuffle).sort_values(['learn type', 'case'])
    df_splitted.to_csv(path_splitted)

    log.add_line('DataFrame path is... ' + path_splitted)
    log.add_line('Done do_split\n')
    return df_splitted


def make_dataset(src_, dataframe_, dst_):
    log.add_line('RUN make_dataset')

    df_train = dataframe_[dataframe_['learn type'] == 'train'].sample(frac=1).reset_index(drop=True)
    df_val = dataframe_[dataframe_['learn type'] == 'val'].sample(frac=1).reset_index(drop=True)
    df_test = dataframe_[dataframe_['learn type'] == 'test']

    util.make_dir(os.path.join(dst_, 'FIESTA', 'train'))
    util.make_dir(os.path.join(dst_, 'FIESTA', 'val'))
    util.make_dir(os.path.join(dst_, 'FIESTA', 'test'))
    util.make_dir(os.path.join(dst_, 'SSFE', 'train'))
    util.make_dir(os.path.join(dst_, 'SSFE', 'val'))
    util.make_dir(os.path.join(dst_, 'SSFE', 'test'))

    df_train_fiesta = df_train[df_train['mri type'] == 'FIESTA']
    df_train_ssfe = df_train[(df_train['mri type'] == 'SSFE') | (df_train['mri type'] == 'T2') | (df_train['mri type'] == 'SSFE(liver&kidney)')]

    df_val_fiesta = df_val[df_val['mri type'] == 'FIESTA']
    df_val_ssfe = df_val[(df_val['mri type'] == 'SSFE') | (df_val['mri type'] == 'T2') | (df_val['mri type'] == 'SSFE(liver&kidney)')]

    df_test_fiesta = df_test[df_test['mri type'] == 'FIESTA']
    df_test_ssfe = df_test[(df_test['mri type'] == 'SSFE') | (df_test['mri type'] == 'T2') | (df_test['mri type'] == 'SSFE(liver&kidney)')]

    # move fiesta
    log.add_line('number of FIESTA train set is... %d' % len(df_train_fiesta.values))
    for i, row in enumerate(df_train_fiesta.values):
        path_src_input = os.path.join(src_, row[0], row[1], 'input', 'input_%03d.npy' % row[3])
        path_src_label = os.path.join(src_, row[0], row[1], 'label', 'label_%03d.npy' % row[3])

        path_dst_input = os.path.join(dst_, 'FIESTA', 'train', 'input_%05d.npy' % i)
        path_dst_label = os.path.join(dst_, 'FIESTA', 'train', 'label_%05d.npy' % i)

        shutil.copy(path_src_input, path_dst_input)
        shutil.copy(path_src_label, path_dst_label)

    log.add_line('number of FIESTA val set is... %d' % len(df_val_fiesta.values))
    for i, row in enumerate(df_val_fiesta.values):
        path_src_input = os.path.join(src_, row[0], row[1], 'input', 'input_%03d.npy' % row[3])
        path_src_label = os.path.join(src_, row[0], row[1], 'label', 'label_%03d.npy' % row[3])

        path_dst_input = os.path.join(dst_, 'FIESTA', 'val', 'input_%05d.npy' % i)
        path_dst_label = os.path.join(dst_, 'FIESTA', 'val', 'label_%05d.npy' % i)

        shutil.copy(path_src_input, path_dst_input)
        shutil.copy(path_src_label, path_dst_label)

    log.add_line('number of FIESTA test set is... %d' % len(df_test_fiesta.values))
    for i, row in enumerate(df_test_fiesta.values):
        path_src_input = os.path.join(src_, row[0], row[1], 'input', 'input_%03d.npy' % row[3])
        path_src_label = os.path.join(src_, row[0], row[1], 'label', 'label_%03d.npy' % row[3])

        path_dst_input = os.path.join(dst_, 'FIESTA', 'test', 'input_%05d.npy' % i)
        path_dst_label = os.path.join(dst_, 'FIESTA', 'test', 'label_%05d.npy' % i)

        shutil.copy(path_src_input, path_dst_input)
        shutil.copy(path_src_label, path_dst_label)

    # move ssfe
    log.add_line('number of SSFE train set is... %d' % len(df_train_ssfe.values))
    for i, row in enumerate(df_train_ssfe.values):
        path_src_input = os.path.join(src_, row[0], row[1], 'input', 'input_%03d.npy' % row[3])
        path_src_label = os.path.join(src_, row[0], row[1], 'label', 'label_%03d.npy' % row[3])

        path_dst_input = os.path.join(dst_, 'SSFE', 'train', 'input_%05d.npy' % i)
        path_dst_label = os.path.join(dst_, 'SSFE', 'train', 'label_%05d.npy' % i)

        shutil.copy(path_src_input, path_dst_input)
        shutil.copy(path_src_label, path_dst_label)

    log.add_line('number of SSFE val set is... %d' % len(df_val_ssfe.values))
    for i, row in enumerate(df_val_ssfe.values):
        path_src_input = os.path.join(src_, row[0], row[1], 'input', 'input_%03d.npy' % row[3])
        path_src_label = os.path.join(src_, row[0], row[1], 'label', 'label_%03d.npy' % row[3])

        path_dst_input = os.path.join(dst_, 'SSFE', 'val', 'input_%05d.npy' % i)
        path_dst_label = os.path.join(dst_, 'SSFE', 'val', 'label_%05d.npy' % i)

        shutil.copy(path_src_input, path_dst_input)
        shutil.copy(path_src_label, path_dst_label)

    log.add_line('number of SSFE test set is... %d' % len(df_test_ssfe.values))
    for i, row in enumerate(df_test_ssfe.values):
        path_src_input = os.path.join(src_, row[0], row[1], 'input', 'input_%03d.npy' % row[3])
        path_src_label = os.path.join(src_, row[0], row[1], 'label', 'label_%03d.npy' % row[3])

        path_dst_input = os.path.join(dst_, 'SSFE', 'test', 'input_%05d.npy' % i)
        path_dst_label = os.path.join(dst_, 'SSFE', 'test', 'label_%05d.npy' % i)

        shutil.copy(path_src_input, path_dst_input)
        shutil.copy(path_src_label, path_dst_label)

    log.add_line('Done make_dataset\n')


main_path_src = os.path.join('./test_datas/datasource')
main_path_dst_pre_split = os.path.join('./pre_split')
main_path_dst = os.path.join('./datasets')


df_pre_split = make_df_pre_split(main_path_src, main_path_dst_pre_split, 'df_pre_split.csv')


print(df_pre_split)

df_splitted = do_split(df_pre_split, 'df_splitted.csv')

make_dataset(main_path_dst_pre_split, df_splitted, main_path_dst)
