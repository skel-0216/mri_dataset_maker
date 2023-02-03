import os
import pandas as pd
import time
import atexit
import numpy as np
import pydicom

bias_index = 4
binary_count = 0
dcm_count = 0

fs_dict = {'COR FIESTA FS BH': True, 'Cor T2 SSFSE BH': False, 'COR T2 SSFSE BH': False, 'COR FIESTA': False,
           'Cor T2 SSFSE FS': True, 'Cor T2 SSFSE FS(LIVER&KIDNEY)': True, 'Cor FIESTA FS BH DL': True,
           'Cor T2 SSFSE BH DL': False, 'Cor T2 SSFSE FS 1': True}


class LogMode:
    def __init__(self, file_path_: str = "../py_log/DefaultFileName", start_time=True):
        start_time = time.strftime('%Y-%m-%d-%H.%M', time.localtime(time.time()))
        if start_time:
            file_path_ = file_path_ + "_" + start_time + ".txt"
        else:
            file_path_ = file_path_ + ".txt"

        self._log_print = True

        self._file_path = file_path_

        self._file = open(file_path_, 'w')

        self.add_line("Start Time : " + start_time + "\n")
        return

    def __del__(self):
        end_time = time.strftime('%Y-%m-%d-%H.%M', time.localtime(time.time()))
        self.add_line("\nEnd Time : " + end_time + "\n")
        self._file.close()

    def mode_print(self, flag=True):
        self._log_print = flag

    def add_line(self, line: str, flag_print: bool = True):
        if flag_print:
            print(line)
        self._file.write(line + "\n")


class DataFrame:
    def __init__(self, file_path_: str = "../DefaultFileName"):
        self.df = pd.DataFrame()
        self.loc_main_dataframe = 0
        self.file_path = file_path_
        atexit.register(self.__close)

    def __close(self):
        start_time = time.strftime('%Y-%m-%d-%H.%M', time.localtime(time.time()))
        file_path_ = os.path.join(self.file_path + "_" + start_time + ".csv")
        self.df.to_csv(file_path_, index=False)

    def set_columns(self, columns_):
        self.df = pd.DataFrame(columns=columns_)

    def add_line(self, line):
        self.df.loc[self.loc_main_dataframe] = line
        self.loc_main_dataframe += 1

    def get_dataframe(self):
        return self.df


def make_dir(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def normalize_npy(image_):
    im1 = image_ - image_.min()
    im2 = im1
    tt = im1.max()

    if tt != 0:
        im2 = (im1 / tt * 255)
        im2.astype('int16')

    return im2


# 파일 내 index 감지용 함수
# bias : 뒤섞여있는 파일들의 인덱스 조정용 bias, %04d 형식이면 2~4 정도로 조정
def find_index(str_list: str, bias=4) -> str:
    index = 0
    str01 = str_list[0]
    str02 = str_list[-1]

    # 인덱스가 0012_악성_... ~ 0042_악성_... 인 경우 인식 안돼서 [i-1:i] 로 범위 변경
    for i in range(len(str01)):
        if str01[i - 1:i] != str02[i - 1:i]:
            if str01[i + 1] == '_':
                index = i
    index -= (bias - 1)
    return index


def do_binary(src_, dst_, normalize=False):
    global binary_count
    binary_count += 1
    print("Now doing do_binary of... ", src_)
    mode_list = os.listdir(src_)

    case = src_[src_.rfind('\\') + 1:]

    for mode_ in mode_list:
        file_list_ = os.listdir(os.path.join(src_, mode_, "binary"))
        length_ = len(os.listdir(os.path.join(src_, mode_, "dcm")))

        # 이중 리스트, 한 요소가 [a, b] 꼴
        # 각 숫자는 binary 데이터 리스트의 순서에 해당함.
        # -1 은 해당 파일에 segment data가 없음을 의미
        index_arr = []

        for i in range(length_):
            index_arr.append([-1, -1])

        cnt = 0
        index = find_index(file_list_)

        for file in file_list_:
            # file page가 0부터 시작이 아닌 1부터 시작으로 보임. 그래서 수정을 위해 1을 빼주었다.
            page = int(file[index:index + bias_index]) - 1
            left_right = int(file[-5])
            # print(page, left_right)  # 인덱스 디버깅 확인용
            index_arr[page][left_right - 1] = cnt
            cnt += 1

        temp_dir_name = os.path.join(dst_, case, mode_, "label")
        if not os.path.exists(temp_dir_name):
            os.makedirs(os.path.join(temp_dir_name))

        for i in range(length_):
            temp_npy = np.zeros((512, 512), dtype='int16')
            if index_arr[i][0] != -1:
                file_name = file_list_[index_arr[i][0]]
                dcm = pydicom.dcmread(os.path.join(src_, mode_, "binary", file_name)).pixel_array
                temp_npy = temp_npy + dcm

            if index_arr[i][1] != -1:
                file_name = file_list_[index_arr[i][1]]
                dcm = pydicom.dcmread(os.path.join(src_, mode_, "binary", file_name)).pixel_array
                temp_npy = temp_npy + dcm

            if normalize:
                np.save(os.path.join(temp_dir_name, "label_%03d" % i), normalize_npy(temp_npy))
            else:
                np.save(os.path.join(temp_dir_name,  "label_%03d" % i), temp_npy)


def filename2number(filename, bias=4):
    return int(filename[-bias-4:-4]) - 1


def do_dcm(src_, dst_, normalize=True):
    global dcm_count
    dcm_count += 1
    print("Now doing do_dcm of... ", src_)
    mode_list = os.listdir(src_)

    case = src_[src_.rfind('\\') + 1:]

    lst_data = []

    for mode_ in mode_list:
        file_list_ = os.listdir(os.path.join(src_, mode_, "dcm"))

        temp_dir_name = os.path.join(dst_, case, mode_, "input")
        if not os.path.exists(temp_dir_name):
            os.makedirs(os.path.join(temp_dir_name))
        for cnt, file in enumerate(file_list_):
            dcm = pydicom.dcmread(os.path.join(src_, mode_, "dcm", file))
            if normalize:
                np.save(os.path.join(temp_dir_name, "input_%03d" % cnt), normalize_npy(dcm.pixel_array))
                lst_data.append([case, mode_, fs_dict[dcm.SeriesDescription], filename2number(file)])
            else:
                np.save(os.path.join(temp_dir_name, "input_%03d" % cnt), dcm.pixel_array)
                lst_data.append([case, mode_, fs_dict[dcm.SeriesDescription], filename2number(file)])
    return lst_data


