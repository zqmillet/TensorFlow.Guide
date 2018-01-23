# -*- coding: utf-8 -*-

from math import exp, log
import numpy
import matplotlib.pyplot

def sigmoid(x, n = 10):
  return 1 / (1 + exp(-n * x)) if x >= 0 else exp(n * x) / (1 + exp(n * x))

def clip(x):
  if (1 - x) ** 2 < 1e-16:
    return 1 - 1e-8
  if x ** 2 < 1e-16:
    return 1e-8
  return x

def l(x1, x2, y, w1, w2):
  return -sum([y * log(clip(sigmoid(z(x1, x2, w1, w2)))) + (1 - y) * log(1 - clip(sigmoid(z(x1, x2, w1, w2)))) for x1, x2, y in zip(x1, x2, y)])

def z(x1, x2, w1, w2):
  return w1 * x1 + w2 * x2 + 1

def get_delta_w1(x1, x2, y, w1, w2):
  return -sum([(y - sigmoid(z(x1, x2, w1, w2))) * x1 for x1, x2, y in zip(x1, x2, y)])

def get_delta_w2(x1, x2, y, w1, w2):
  return -sum([(y - sigmoid(z(x1, x2, w1, w2))) * x2 for x1, x2, y in zip(x1, x2, y)])

def main():
  with open('training_data.dat', 'r') as data_file:
    next(data_file)
    data = list(zip(*[[float(x.strip()) for x in line.split(',')] for line in data_file if not line.strip() == '']))


  x1, x2, y = (column for column in data)
  w1, w2= 0, 0
  r = 0.06

  precision = 1e-4

  parameters = []
  index = 0
  while True:
    delta_w1, delta_w2 = get_delta_w1(x1, x2, y, w1, w2), get_delta_w2(x1, x2, y, w1, w2)
    if delta_w1**2 + delta_w2**2 < precision**2:
      break

    parameters.append([w1, w2, l(x1, x2, y, w1, w2)])
    w1, w2 = w1 - r * delta_w1, w2 - r * delta_w2

    index += 1
    if index > 20:
      break

  with open('parameters_n_equal_10_big_learning_rate.dat', 'w', encoding = 'utf8') as parameters_file:
    parameters_file.write('w1, w2, l\n')
    for parameter in parameters:
      parameters_file.write(', '.join([str(item) for item in parameter]) + '\n')
  print('w1 = {w1}, w2 = {w2}'.format(w1 = w1, w2 = w2))

  for x1_, x2_, y_ in zip(data[0], data[1], data[2]):
    z_ = 1 if z(x1_, x2_, w1, w2) > 0 else 0
    if not z_ == y_:
      print(x1_, x2_)


  # 绘制函数 $L(\bm{W}, 1)$ 等高线
  # 首选获取参数 $W_{11}^{(1)}$ 和 $W_{12}^{(1)}$ 的范围
  w1_max = max([x[0] for x in parameters])
  w1_min = min([x[0] for x in parameters])
  w2_max = max([x[1] for x in parameters])
  w2_min = min([x[1] for x in parameters])

  # 设置采样密度
  w1_sample_number = 50
  w2_sample_number = 50

  # 参数 $W_{11}^{(1)}$ 和 $W_{12}^{(1)}$ 网格化
  w1_samples = numpy.linspace(-2, 4, w1_sample_number)
  w2_samples = numpy.linspace(w2_min, w2_max, w2_sample_number)
  w1_samples, w2_samples = numpy.meshgrid(w1_samples, w2_samples)

  # 计算网格化的 $L(\bm{W}, 1)$ 结果
  l_samples = [[0]*w1_sample_number for _ in range(w2_sample_number)]
  for i in range(w1_sample_number):
    for j in range(w2_sample_number):
      l_samples[j][i] = l(x1, x2, y, w1_samples[j][i], w2_samples[j][i])

  l_max = max([max(x) for x in l_samples])
  l_min = min([min(x) for x in l_samples])
  l_samples = numpy.array(l_samples)

  # 设置等高线高度
  contour_level = numpy.fix(l_min) + 1
  contour_levels = []
  delta_level = 1
  while contour_level < l_max:
    contour_levels.append(contour_level)
    contour_level += delta_level
    delta_level *= 2

  # 获取等高线数据
  contour_data = matplotlib.pyplot.contour(w1_samples, w2_samples, l_samples, levels = contour_levels)
  matplotlib.pyplot.show()

  # 将等高线数据保存在文件中
  with open('./contour_n_equal_10_big_learning_rate.dat', 'w') as contour_data_file:
    # import pdb; pdb.set_trace()
    for i in range(len(contour_data.collections)):
      for j in range(len(contour_data.collections[i].get_paths())):
        for k in range(len(contour_data.collections[i].get_paths()[j].vertices)):
          coordinate = contour_data.collections[i].get_paths()[j].vertices[k]
          contour_data_file.write('{0}, {1}, {2}\n'.format(coordinate[0], coordinate[1], contour_data.layers[i]))
        contour_data_file.write('\n')

if __name__ == '__main__':
    main()