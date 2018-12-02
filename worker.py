# coding=utf-8
from __future__ import print_function
from data_convert import load_student_data
from data_convert import load_label
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
import numpy as np

x_raw = load_student_data()
y_raw = load_label()

x_pair = []
y_pair = []

for index in range(len(y_raw)):
    for i in range(1, len(y_raw) - index):
        if y_raw[index] > y_raw[index + i]:
            y_pair.append(1)
        else:
            y_pair.append(0)
print("y_pair build finish shape: %s" % len(y_pair))

for index in range(len(x_raw)):
    for i in range(1, len(x_raw) - index):
        x_pair.append(x_raw[index] - x_raw[index + i])
print("x_pair build finish shape: %s" % len(x_pair))
x = np.array(x_pair)
y = np.array(y_pair)
# np.save('./data/x.npy', x)
# np.save('./data/y.npy', y)
# x = np.load('./data/x.npy')
# y = np.load('./data/y.npy')
print("x set shape :", x.shape, "y set shape :", y.shape)

rate = int(x.shape[0] * 0.7)
x_train = x[0:rate]
x_test = x[rate:]
y_train = y[0:rate]
y_test = y[rate:]

print("x_train", x_train.shape, "x_test", x_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
print("----------------------打印训练集-----------------------------------")
# for index in range(x_train.shape[0]):
#     print("x: %s | y: %s" % (x_train[index], y_train[index]))
print("----------------------训练集打印完毕-------------------------------")


def deep():
    print("深度学习方法实现")
    # 网络搭建
    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], units=1, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(0.8))
    model.add(Activation('relu'))
    model.add(Dense(30))
    model.add(Dropout(0.8))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam', metrics=["accuracy"], loss='binary_crossentropy')
    model.fit(x_train, y_train, batch_size=16, epochs=5)
    score = model.evaluate(x_test, y_test, batch_size=16)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    W, b = model.layers[0].get_weights()

    print('Weights=', W, '\n biases=', b)

    # plotting the prediction
    y_pred = model.predict(x_test)
    # print("raw y_pred")
    # print(y_pred)
    for mindex in y_pred:
        mindex[0] = round(mindex[0], 0)

    print("--------------打印预测结果与实际值--------------")
    # y_pred = y_pred.astype(int)
    # for mindex in zip(y_test, y_pred):
    #     print("y_test", mindex[0], "| y_pred", mindex[1])
    #
    print("--------------预测结果与实际值打印完毕-----------")

deep()


def linear_regression():
    print("线性回归方法")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score

    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    print(linreg.intercept_)
    print(linreg.coef_)

    y_pred = linreg.predict(x_test)

    # for t in zip(y_test, y_pred):
    #     print("y_test", t[0], "| y_pred", round(t[1]))

    sum_mean = 0
    for ia in range(len(y_pred)):
        sum_mean += (y_pred[ia] - y_test[ia]) ** 2
    sum_erro = np.sqrt(sum_mean / 50)
    print("RMSE by hand:", sum_erro)
    print("accuracy: %s" % accuracy_score(y_test, y_pred))


def svm_class():
    print("支持向量机方法")
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # for t in zip(y_test, y_pred):
    #     print("y_test", t[0], "| y_pred", round(t[1]))
    print("正在评估性能")
    sum_mean = 0
    for index_i in range(len(y_pred)):
        sum_mean += (y_pred[index_i] - y_test[index_i]) ** 2
    sum_erro = np.sqrt(sum_mean / 50)
    print("RMSE by hand:", sum_erro)
    print("accuracy: %s" % accuracy_score(y_test, y_pred))


svm_class()
