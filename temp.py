# coding=utf-8
import sys

a = [5, 2, 1, 4, 6, 8, 9, 11, 3]


def compare(element_a):
    for index in range(len(element_a)):
        for i in range(1, len(element_a) - index):
            print("%s > %s" % (element_a[index], element_a[index + i]), element_a[index] > element_a[index + i])


# compare(a)


import numpy as np

# xxx = np.genfromtxt('./data/consumption.txt', dtype=str)[1:]
# # ('period', 'i8'), ('number', 'i8'), ('place', 's5'), ('date', 'i8'), ('time', 'i8'),('money', 'f8')
# print(xxx[1:100])


# np.save('./data/borrow.npy', xxx)

def consumpution_al():
    result = []
    xxx = np.genfromtxt('./data/consumption.txt', dtype=str)[1:]
    for index in xxx:
        result.append([int(index[1]), str(index[2]), str(index[3]), int(index[4]), float(index[5])])
    return result


# xxxx = np.load('./data/access.npy')
# print(xxxx)

def catlll():
    catalog = []
    if sys.version > '3':
        catalog = np.load('./data/catalog.npy')
        book_catalog = []
        for index in range(len(catalog)):
            book_catalog.append([catalog[index][0].decode(), catalog[index][1].decode()])
    else:
        catalog = np.load('./data/catalog.npy')
        # print(catalog)


con = consumpution_al()
print(con)
