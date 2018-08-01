from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from utils.data_gmm import sample_GMM
from utils.data_utils import shuffle, iter_data
from tqdm import tqdm
import os
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


""" parameters """
n_epoch = 200
batch_size = 128
dataset_size_x = 512*4
dataset_size_e1 = 512*4
dataset_size_e2 = 512*4
dataset_size_y = 512*4

dataset_size_e2_test = 512*2
dataset_size_e1_test = 512*2
dataset_size_x_test = 512*2
dataset_size_y_test = 512*2
x_dim = 2
y_dim = 2
latent_dim = 2
eps_dim = 2

n_layer_disc = 2
n_hidden_disc = 256
n_layer_gen = 2
n_hidden_gen = 256
n_layer_inf = 2
n_hidden_inf = 256

""" Create directory for results """
result_dir = 'results/'
directory = result_dir
if not os.path.exists(directory):
    os.makedirs(directory)


""" Create dataset """

# create X dataset
means_x = map(lambda x:  np.array(x), [[0, 0],
                                       [2, 2],
                                       [-2, -2],
                                       [2, -2],
                                       [-2, 2]])
means_x = list(means_x)
std_x = 0.04
variances_x = [np.eye(2) * std_x for _ in means_x]

priors_x = [1.0/len(means_x) for _ in means_x]
dataset_x = sample_GMM(dataset_size_x, means_x, variances_x, priors_x, sources=('features', ))
save_path_x = result_dir + 'X_train.png'

# plot x
X_dataset = dataset_x.data['samples']
X_targets = dataset_x.data['label']

fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(X_dataset[:, 0], X_dataset[:, 1], c=cm.Set1(X_targets.astype(float)/x_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x$ train')
ax.axis('on')
plt.savefig(save_path_x, transparent=True, bbox_inches='tight')

# create Y dataset
means_y = map(lambda x:  np.array(x), [[0, 2],
                                       [2, 0],
                                       [0, 0],
                                       [-2, 0],
                                       [0, -2]])
means_y = list(means_y)
std_y = 0.04
variances_y = [np.eye(2) * std_y for _ in means_y]

priors_y = [1.0/len(means_y) for _ in means_y]
dataset_y = sample_GMM(dataset_size_y, means_y, variances_y, priors_y, sources=('features', ))
save_path_y = result_dir + 'Y_train.png'

# plot y
Y_dataset = dataset_y.data['samples']
Y_targets = dataset_y.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(Y_dataset[:, 0], Y_dataset[:, 1], c=cm.Set1(Y_targets.astype(float)/y_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$y$ train')
ax.axis('on')
plt.savefig(save_path_y, transparent=True, bbox_inches='tight')


# create eplison1 train dataset
means_e1 = map(lambda x:  np.array(x), [[0, 0]])
means_e1 = list(means_e1)
std_e1 = 1.0
variances_e1 = [np.eye(2) * std_e1 for _ in means_e1]
priors_e1 = [1.0/len(means_e1) for _ in means_e1]

dataset_e1 = sample_GMM(dataset_size_e1, means_e1, variances_e1, priors_e1, sources=('features', ))
save_path_e1 = result_dir + 'e1_train.png'

#  plot epsilon1
e1_dataset = dataset_e1.data['samples']
e1_labels = dataset_e1.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(e1_dataset[:, 0], e1_dataset[:, 1], c='r',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$e1$ train')
ax.axis('on')
plt.savefig(save_path_e1, transparent=True, bbox_inches='tight')

# create eplison2 train dataset
means_e2 = map(lambda x: np.array(x), [[0, 0]])
means_e2 = list(means_e2)
std_e2 = 1.0
variances_e2 = [np.eye(2) * std_e2 for _ in means_e2]
priors_e2 = [1.0/len(means_e2) for _ in means_e2]

dataset_e2 = sample_GMM(dataset_size_e2, means_e2, variances_e2, priors_e2, sources=('features', ))
save_path_e2 = result_dir + 'e2_train.png'

#  plot epsilon2
e2_dataset = dataset_e2.data['samples']
e2_labels = dataset_e2.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(e2_dataset[:, 0], e2_dataset[:, 1], c='r',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$e2$ train')
ax.axis('on')
plt.savefig(save_path_e2, transparent=True, bbox_inches='tight')


""" Networks """


def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs)),  tf.float32)


def x_generator(e, x_dim, n_layer, n_hidden):
    with tf.variable_scope("x_generator"):
        h = slim.repeat(e, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, x_dim, activation_fn=None, scope="p_x")
    return x


def y_generator(e, y_dim, n_layer, n_hidden):
    with tf.variable_scope("y_generator"):
        h = slim.repeat(e, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        y = slim.fully_connected(h, y_dim, activation_fn=None, scope="p_y")
    return y


def generator_x2y(x, y_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("generator_x2y"):
        eps = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([x, eps], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        y = slim.fully_connected(h, y_dim, activation_fn=None)
    return y


def generator_y2x(y, x_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("generator_y2x"):
        eps = standard_normal([y.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([y, eps], 1)
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, x_dim, activation_fn=None)
    return x


def discriminator(x, y, n_layers=2, n_hidden=256, activation_fn=None):
    h = tf.concat([x, y], 1)
    with tf.variable_scope('discriminator'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 4, activation_fn=activation_fn)
    return log_d


""" Construct model and training ops """
tf.reset_default_graph()

e1 = tf.placeholder(tf.float32, shape=(batch_size, latent_dim))
e2 = tf.placeholder(tf.float32, shape=(batch_size, latent_dim))
x = tf.placeholder(tf.float32, shape=(batch_size, x_dim))
y = tf.placeholder(tf.float32, shape=(batch_size, y_dim))

# generator
# p related to generative samples, q related to real samples
p_x = x_generator(e1, x_dim, n_layer_gen, n_hidden_gen)
p_y = y_generator(e2, y_dim, n_layer_gen, n_hidden_gen)
p_x2y = generator_x2y(p_x, y_dim, n_layer_gen, n_hidden_gen, eps_dim)
p_y2x = generator_y2x(p_y, x_dim, n_layer_gen, n_hidden_gen, eps_dim)

q_x2y = generator_x2y(x, y_dim, n_layer_gen, n_hidden_gen, eps_dim)
q_y2x = generator_y2x(y, x_dim, n_layer_gen, n_hidden_gen, eps_dim)

rec_x = generator_y2x(q_x2y, x_dim, n_layer_gen, n_hidden_gen, eps_dim)
rec_y = generator_x2y(q_y2x, y_dim, n_layer_gen, n_hidden_gen, eps_dim)

p1_logit = discriminator(x, q_x2y, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
p2_logit = discriminator(q_y2x, y, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
p3_logit = discriminator(p_x, p_x2y, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
p4_logit = discriminator(p_y2x, p_y, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

p1_label = tf.one_hot(tf.fill([int(p1_logit.get_shape()[0]), 1], 0), 4)
p1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p1_label, logits=p1_logit))

p2_label = tf.one_hot(tf.fill((int(p2_logit.get_shape()[0]), 1), 1), 4)
p2_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p2_label, logits=p2_logit))

p3_label = tf.one_hot(tf.fill((int(p3_logit.get_shape()[0]), 1), 2), 4)
p3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p3_label, logits=p3_logit))

p4_label = tf.one_hot(tf.fill([int(p4_logit.get_shape()[0]), 1], 3), 4)
p4_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p4_label, logits=p4_logit))

disc_loss = p1_loss + p2_loss + p3_loss + p4_loss

cost_y = tf.reduce_mean(tf.pow(rec_y - y, 2))
cost_x = tf.reduce_mean(tf.pow(rec_x - x, 2))

gen_loss = -disc_loss + cost_x + cost_y

px_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "x_generator")
py_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "y_generator")
x2y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator_x2y")
y2x_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator_y2x")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)
train_gen_op = opt.minimize(gen_loss, var_list=px_vars + py_vars + x2y_vars + y2x_vars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars)

""" training """
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


FG = []
FD = []

for epoch in tqdm(range(n_epoch), total=n_epoch):
    X_dataset= shuffle(X_dataset)
    Y_dataset = shuffle(Y_dataset)
    e1_dataset= shuffle(e1_dataset)
    e2_dataset = shuffle(e2_dataset)
    i = 0
    for xmb, ymb, e1mb, e2mb in iter_data(X_dataset, Y_dataset, e1_dataset, e2_dataset, size=batch_size):
        i = i + 1
        for _ in range(1):
            f_d, _ = sess.run([disc_loss, train_disc_op], feed_dict={x: xmb, y:ymb, e1:e1mb, e2:e2mb})
        for _ in range(1):
            f_g, _ = sess.run([[gen_loss, cost_x, cost_y], train_gen_op], feed_dict={x: xmb, y:ymb, e1:e1mb, e2:e2mb})
        FG.append(f_g)
        FD.append(f_d)

    print("epoch %d iter %d: discloss %f genloss %f recons_x %f recons_y %f"
          %(epoch, i, f_d, f_g[0], f_g[1], f_g[2]))

""" plot the results """

# test dataset

# create epsilon1 dataset
e1_test = sample_GMM(dataset_size_e1_test, means_e1, variances_e1, priors_e1, sources=('features', ))
save_path = result_dir + 'e1_test.png'

#  plot epsilon1_test
e1_data_test = e1_test.data['samples']
e1_targets_test = e1_test.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(e1_data_test[:, 0], e1_data_test[:, 1], c='r', edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$e1$ test')
ax.axis('on')
plt.savefig(save_path, transparent=True, bbox_inches='tight')

# create epsilon2 test dataset
e2_test = sample_GMM(dataset_size_e2_test, means_e2, variances_e2, priors_e2, sources=('features', ))
save_path = result_dir + 'e2_test.png'

#  plot epsilon2_test
e2_data_test = e2_test.data['samples']
e2_targets_test = e2_test.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(e2_data_test[:, 0], e2_data_test[:, 1], c='r', edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$e2$ test')
ax.axis('on')
plt.savefig(save_path, transparent=True, bbox_inches='tight')

# create X test dataset
dataset_x_test = sample_GMM(dataset_size_x_test, means_x, variances_x, priors_x, sources=('features', ))
save_path_x = result_dir + 'X_test.png'

# plot x_test
X_dataset_test = dataset_x_test.data['samples']
X_targets_test = dataset_x_test.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(X_dataset_test[:, 0], X_dataset_test[:, 1], c=cm.Set1(X_targets_test.astype(float)/x_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x$ test')
ax.axis('on')
plt.savefig(save_path_x, transparent=True, bbox_inches='tight')

# create Y test dataset
dataset_y_test = sample_GMM(dataset_size_y_test, means_y, variances_y, priors_y, sources=('features', ))
save_path_y = result_dir + 'Y_test.png'

# plot y_test
Y_dataset_test = dataset_y_test.data['samples']
Y_targets_test = dataset_y_test.data['label']

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(Y_dataset_test[:, 0], Y_dataset_test[:, 1], c=cm.Set1(Y_targets_test.astype(float)/y_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$y$ test')
ax.axis('on')
plt.savefig(save_path_y, transparent=True, bbox_inches='tight')

n_viz = 1
x_margin = np.array([]); y_margin = np.array([]); p_x2y_test = np.array([]); p_y2x_test = np.array([])
condition_x = np.array([]); condition_y = np.array([])
rec_x = np.array([]); rec_y = np.array([])

mse_x = 0
mse_y = 0
for _ in range(n_viz):
    for e1mb, e2mb, xmb, ymb in iter_data(e1_data_test, e2_data_test, X_dataset_test, Y_dataset_test, size=batch_size):
        temp_x = sess.run(p_x, feed_dict={e1: e1mb})
        x_margin = np.vstack([x_margin, temp_x]) if x_margin.size else temp_x

        temp_y = sess.run(p_y, feed_dict={e2: e2mb})
        y_margin = np.vstack([y_margin, temp_y]) if y_margin.size else temp_y

        tmp_p_xy = sess.run(p_x2y, feed_dict={p_x: temp_x})
        p_x2y_test = np.vstack([p_x2y_test, tmp_p_xy]) if p_x2y_test.size else tmp_p_xy

        tmp_p_yx = sess.run(p_y2x, feed_dict={p_y: temp_y})
        p_y2x_test = np.vstack([p_y2x_test, tmp_p_yx]) if p_y2x_test.size else tmp_p_yx

        tmp_cy = sess.run(q_x2y, feed_dict={x:xmb})
        condition_y = np.vstack([condition_y, tmp_cy]) if condition_y.size else tmp_cy

        tmp_cx = sess.run(q_y2x, feed_dict={y: ymb})
        condition_x = np.vstack([condition_x, tmp_cx]) if condition_x.size else tmp_cx

        tmp_recx = sess.run(p_y2x, feed_dict={p_y:tmp_cy})
        rec_x = np.vstack([rec_x, tmp_recx]) if rec_x.size else tmp_recx

        tmp_recy = sess.run(p_x2y, feed_dict={p_x: tmp_cx})
        rec_y = np.vstack([rec_y, tmp_recy]) if rec_y.size else tmp_recy

        mse_y += np.sum((tmp_recy - ymb)**2)
        mse_x += np.sum((tmp_recx - xmb) ** 2)
mse_x /= X_dataset_test.shape[0]
mse_y /= Y_dataset_test.shape[0]

# plot marginal x
fig_mz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(x_margin[:, 0], x_margin[:, 1], c='b', edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x$ generated from $e1$')
ax.axis('on')
plt.savefig(result_dir + 'marginal_x_test.png', transparent=True, bbox_inches='tight')

#  plot marginal y
fig_pz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(y_margin[:, 0], y_margin[:, 1], c='g',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$y$ generated from $e2$')
ax.axis('on')
plt.savefig(result_dir + 'marginal_y_test.png', transparent=True, bbox_inches='tight')

# p(y|x)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(p_x2y_test[:, 0], p_x2y_test[:, 1], c='g',
        edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$y$ generated from fake $x$')
ax.axis('on')
plt.savefig(result_dir + 'conditional_y_test.png', transparent=True, bbox_inches='tight')

# p(x|y)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(p_y2x_test[:, 0], p_y2x_test[:, 1], c='b',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x$ generated from fake $y$')
ax.axis('on')
plt.savefig(result_dir + 'conditional_x_test.png', transparent=True, bbox_inches='tight')

# x_real to y
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(condition_y[:, 0], condition_y[:, 1], c='b',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$y$ generated from real $x$')
ax.axis('on')
plt.savefig(result_dir + 'x_real_to_y_test.png', transparent=True, bbox_inches='tight')

# y_real to x
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(condition_x[:, 0], condition_x[:, 1], c='b',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x$ generated from real $y$')

ax.axis('on')
plt.savefig(result_dir + 'y_real_to_x_test.png', transparent=True, bbox_inches='tight')

# reconstruct x test
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(rec_x[:, 0], rec_x[:, 1], c='b',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('reconstructed $x$')
ax.axis('on')
plt.savefig(result_dir + 'rec_x_test.png', transparent=True, bbox_inches='tight')

# reconstruct y test
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(rec_y[:, 0], rec_y[:, 1], c='b',
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('reconstructed $y$')
ax.axis('on')
plt.savefig(result_dir + 'rec_y_test.png', transparent=True, bbox_inches='tight')

# learning curves
fig_curve, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.plot(FD, label="Discriminator")
ax.plot(np.array(FG)[:, 0], label="Generator")
ax.plot(np.array(FG)[:, 1], label="Reconstruction X")
ax.plot(np.array(FG)[:, 2], label="Reconstruction Y")
plt.xlabel('Iteration')
plt.ylabel('Loss')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.axis('on')
plt.savefig(result_dir + 'learning_curves.png', bbox_inches='tight')

# mse
print('MSE of X: %f' % mse_x)
print('MSE of Y: %f' % mse_y)
