import numpy as np
from TrainingSet import TrainGAN

mean_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
cov_x = np.random.rand(10, 10)
cov_x = np.identity(10) + cov_x + cov_x.T
num_x = 100

mean_gen = np.array([0, 0, 0])
cov_gen = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

x = np.random.multivariate_normal(mean_x, cov_x, num_x)
x = [x[c] for c in range(x.shape[0])]

num_class = 1
size_batch = 20
test_size = 0.1
valid_size = 0.1
num_layer = 2
max_num_iter = 1000
valid_epoch = 2

GAN = TrainGAN(x, mean_gen, cov_gen, num_class=num_class,
               size_batch=size_batch, test_size=test_size,
               valid_size=valid_size, num_layer=num_layer,
               max_num_iter=max_num_iter, valid_epoch=valid_epoch)

GAN.train(10)
