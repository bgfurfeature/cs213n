import matplotlib.pyplot as plt

# Load the (preprocessed) data.
# def load_data(filename):
#     f = open(filename, "r")
#     data = []
#     lines = f.readlines()
#     for item in lines:
#         data.append(item.split("\t")[1])
#     f.close()
#
#     return data
#
#
# file1 = load_data("F:\\datatest\\crawlerTest\\info_log\\file1_70")
# file2 = load_data("F:\\datatest\\crawlerTest\\info_log\\file1_78")
# file21 = load_data("F:\\datatest\\crawlerTest\\info_log\\70_info.log_start")
# file22 = load_data("F:\\datatest\\crawlerTest\\info_log\\70_info.log_done")
# file31 = load_data("F:\\datatest\crawlerTest\info_log\\78_info.log_start")
# file32 = load_data("F:\\datatest\\crawlerTest\\info_log\\78_info.log_done")
# file41 = load_data("F:\\datatest\\crawlerTest\\info_log\\timePath_70")
# file42 = load_data("F:\\datatest\\crawlerTest\\info_log\\timePath_78")
# plt.subplot(2, 2, 1)
# plt.plot(file1, 'o')
# plt.legend(['send_time'], loc='upper left')
# plt.xlabel('x')
# plt.ylabel('timeStamp')
# # 70
# plt.subplot(2, 2, 2)
# plt.title("server 70")
# plt.plot(file1, 'o')
# plt.plot(file21, '-o')
# plt.plot(file22, '-o')
# plt.legend(['send_time','start_time_70', 'stop_time'], loc='upper left')
# plt.xlabel('x')
# plt.ylabel('timeStamp')
# # 78
# plt.subplot(2, 2, 3)
# plt.title("server 78")
# plt.plot(file1, 'o')
# plt.plot(file31, '-o')
# plt.plot(file32, '-o')
# plt.legend(['send_time','start_time_78', 'stop_time'], loc='upper left')
# plt.xlabel('x')
# plt.ylabel('timeStamp')
#
# plt.subplot(2, 2, 4)
# plt.plot(file41, '-o')
# plt.plot(file42, '-o')
# plt.legend(['hbase_time_70','hbase_time_78'], loc='upper left')
# plt.xlabel('x')
# plt.ylabel('timeStamp')
#
# plt.show()
x = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3,
     4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
     2, 3,
     4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
     2, 3,
     4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
     2, 3,
     4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
     2, 3,
     4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
     2, 3,
     4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
     2, 3,
     4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,
     2, 3,
     4, 5]

y = [3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3,
     5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3,
     5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3,
     5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3,
     5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3,
     5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3,
     5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3,
     5, 7, 9, 11,
     3, 5, 7, 9, 11, 3, 5, 7, 9, 11]
a = 0
b = 0


def forward(x, a, b):
    return a * x + b


def loss(x_, a, b):
    loss_local = 0
    for xi in range(0, len(x)):
        loss_local += pow(forward(x[xi], a, b), 2) / 2
    return loss_local


def gradient(x, a, b):
    for xi in range(0, len(x)):
        h = forward(x[xi], a, b)
        da = l * ((y[xi] - h) * x[xi])
        db = l * (y[xi] - h)
        a += da
        b += db
    g = (a, b)
    return g


opti = 0
l = 1e-5
print len(x)
print len(y)
for i in xrange(10000):
    if i == 0:
        opti = loss(x, a, b)
        # print opti

    loss_ = loss(x, a, b)

    if opti > loss_:
        opti = loss_

    c = gradient(x, a, b)
    (a, b) = c

for i in x:
    print forward(i, a, b)
print a, b
