# coding=utf-8
import numpy as np
# import sklearn as skl
import sys
from sklearn import preprocessing



def split_period(file_name):
    first_period = []
    second_period = []
    third_period = []
    raw = np.genfromtxt(file_name)[1:]
    # raw = np.load(file_name)
    for x in range(len(raw)):
        if raw[x][0] == 1:
            first_period.append(raw[x][1:])
        if raw[x][0] == 2:
            second_period.append(raw[x][1:])
        if raw[x][0] == 3:
            third_period.append(raw[x][1:])
    first_period = np.array(first_period)
    second_period = np.array(second_period)
    third_period = np.array(third_period)
    return first_period, second_period, third_period


def split_term_consumption(file_name):
    first_period = []
    second_period = []
    third_period = []
    data = np.genfromtxt(file_name, dtype=str)[1:]
    for index in data:
        if index[0] == '1':
            # 学号 地点 日期 时间 金额
            first_period.append([int(index[1]), str(index[2]), str(index[3]), int(index[4]), float(index[5])])
        elif index[0] == '2':
            second_period.append([int(index[1]), str(index[2]), str(index[3]), int(index[4]), float(index[5])])
        elif index[0] == '3':
            third_period.append([int(index[1]), str(index[2]), str(index[3]), int(index[4]), float(index[5])])
    # first_period = np.array(first_period)
    # second_period = np.array(second_period)
    # third_period = np.array(third_period)
    return first_period, second_period, third_period


def split_term_consumption_npy(file_name):
    first_period = []
    second_period = []
    third_period = []
    data = np.load(file_name)[1:]
    for index in data:
        if index[0] == '1':
            # 学号 地点 日期 时间 金额
            first_period.append([int(index[1]), str(index[2]), str(index[3]), int(index[4]), float(index[5])])
        elif index[0] == '2':
            second_period.append([int(index[1]), str(index[2]), str(index[3]), int(index[4]), float(index[5])])
        elif index[0] == '3':
            third_period.append([int(index[1]), str(index[2]), str(index[3]), int(index[4]), float(index[5])])
    # first_period = np.array(first_period)
    # second_period = np.array(second_period)
    # third_period = np.array(third_period)
    return first_period, second_period, third_period

def counter(larray):
    counters = []
    for x in np.arange(1, 539, 1):
        y = larray == x
        z = larray[y]
        counters.append((x, z.size))
    return np.array(counters)


def counter_health_eating_consumption(consumption):
    money_counters = []
    for x in np.arange(1, 539, 1):
        money_counters.append([x, 0])
    for index in consumption:
        # print("student id :",int(index[0]))
        # print("money_counters[int(index[0])]",money_counters[int(index[0])-1])
        if index[1] == '\xca\xb3\xcc\xc3':
            # 三餐均规律
            # if index[3] > 103000 < 123000 or index[3] > 63000 < 83000 or index[3] > 164000 < 183000:
            if index[3] > 63000 < 83000:
                # 三餐按时吃的金额
                # money_counters[int(index[0]) - 1][1] += index[4]
                # 三餐按时吃的频次
                money_counters[int(index[0]) - 1][1] += 1
    return np.array(money_counters)


def counter__consumption(consumption):
    money_counters = []
    for x in np.arange(1, 539, 1):
        money_counters.append([x, 0])
    for index in consumption:
        # print("student id :",int(index[0]))
        # print("money_counters[int(index[0])]",money_counters[int(index[0])-1])
        money_counters[int(index[0]) - 1][1] += index[4]
    return np.array(money_counters)


def catalog2list():
    catalog = []
    if sys.version > '3':
        book_catalog = np.load('./data/catalog.npy')
        for index in range(len(book_catalog)):
            catalog.append([book_catalog[index][0].decode(), book_catalog[index][1].decode()])
    else:
        # catalog = np.genfromtxt('./data/catalog.txt', dtype='str')[1:]\
        catalog = np.load('./data/catalog.npy')
    return catalog


def counter_book(book_list):
    book = dict()
    for x in book_list:
        if book.get(x[1]) is not None:
            book[x[1]] += 1
        else:
            book[x[1]] = 1
    return book


score_1st, score_2nd, score_3rd = split_period('./data/score.txt')
access_1st, access_2nd, access_3rd = split_period('./data/access.txt')
# consumption_1st, consumption_2nd, consumption_3rd = split_term_consumption('./data/consumption.txt')  # python2

consumption_1st, consumption_2nd, consumption_3rd = split_term_consumption_npy('./data/consumption.npy')  # python3
borrow_1st, borrow_2nd, borrow_3rd = split_period('./data/borrow.txt')

# 处理图书分类
lib_list = catalog2list()
lib_dict = dict(lib_list)
# print(lib_list)
# print("book status")
# # 不同书类别的数量
# book_dict = counter_book(lib_list)
# book_label = book_dict.keys()
# print(book_dict)
# print("book_label")
# # 书类别标签
# print(book_label)
print("")
# 第三学期学生借书频次记录
borrow_3rd_time = counter(np.sort(borrow_3rd[:, 0]))
# 第三学期学生进入图书馆频次记录
access_3rd_times = counter(np.sort(access_3rd[:, 0]))
# 第三学期学生综合排名记录
score_3rd_times = score_3rd[score_3rd[:, 0].argsort()]
# 第三学期学生健康饮食记录
health_eat_consumption_3rd_times = counter_health_eating_consumption(consumption_3rd)
# 第三学期学生消费总额
consumption_3rd_times = counter__consumption(consumption_3rd)
# 第一学期学生综合排名记录
score_1st_times = score_1st[score_1st[:, 0].argsort()]
# 第二学期学生综合排名记录
score_2nd_times = score_2nd[score_1st[:, 0].argsort()]
print("3rd period borrow")
print(borrow_3rd_time)
print("3rd period score")
print(score_3rd_times)
print("3rd period lib access times")
print(access_3rd_times)
print("3rd period consumption")
print(health_eat_consumption_3rd_times)
print("lib book catalog")
print(lib_dict.get('1000201'))

print(" ")




def load_student_data():
    student_data = []
    for x in np.arange(538):
        student_data.append(
            [borrow_3rd_time[x][1], access_3rd_times[x][1], consumption_3rd_times[x][1],
             health_eat_consumption_3rd_times[x][1], score_1st_times[x][1],
             score_2nd_times[x][1]])
    return preprocessing.scale(np.array(student_data))


def load_label():
    return score_3rd_times[:, 1]


