#!/usr/bin/env python3

import tensorflow as tf

class ReLoBRaLoLoss(tf.keras.losses.Loss):

    def __init__(self, model, alpha=0.999, temperature=0.1, rho=0.99, name='ReLoBRaLoLoss', **kwargs):
        super(ReLoBRaLoLoss, self).__init__(name=name, **kwargs)
        self.model = model
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.call_count = tf.Variable(0, trainable=False, dtype=tf.int16)

        self.numTerms = len(model.alpha) + len(model.beta) + len(model.gamma)
        self.lambdas = [tf.Variable(1., trainable=False) for _ in range(self.numTerms)]
        self.last_losses = [tf.Variable(1., trainable=False) for _ in range(self.numTerms)]
        self.init_losses = [tf.Variable(1., trainable=False) for _ in range(self.numTerms)]

    def call(self, losses, dummy):
        EPS = 1e-7

        # in first iteration (self.call_count == 0), drop lambda_hat and use init lambdas, i.e. lambda = 1
        #   i.e. alpha = 1 and rho = 1
        # in second iteration (self.call_count == 1), drop init lambdas and use only lambda_hat
        #   i.e. alpha = 0 and rho = 1
        # afterwards, default procedure (see paper)
        #   i.e. alpha = self.alpha and rho = Bernoully random variable with p = self.rho
        alpha = tf.cond(tf.equal(self.call_count, 0),
                lambda: 1.,
                lambda: tf.cond(tf.equal(self.call_count, 1),
                                lambda: 0.,
                                lambda: self.alpha))
        rho = tf.cond(tf.equal(self.call_count, 0),
              lambda: 1.,
              lambda: tf.cond(tf.equal(self.call_count, 1),
                              lambda: 1.,
                              lambda: tf.cast(tf.random.uniform(shape=()) < self.rho, dtype=tf.float32)))

        # compute new lambdas w.r.t. the losses in the previous iteration
        lambdas_hat = [losses[i] / (self.last_losses[i] * self.temperature + EPS) for i in range(len(losses))]
        lambdas_hat = tf.nn.softmax(lambdas_hat - tf.reduce_max(lambdas_hat)) * tf.cast(len(losses), dtype=tf.float32)

        # compute new lambdas w.r.t. the losses in the first iteration
        init_lambdas_hat = [losses[i] / (self.init_losses[i] * self.temperature + EPS) for i in range(len(losses))]
        init_lambdas_hat = tf.nn.softmax(init_lambdas_hat - tf.reduce_max(init_lambdas_hat)) * tf.cast(len(losses), dtype=tf.float32)
        
        # use rho for deciding, whether a random lookback should be performed
        new_lambdas = [(rho * alpha * self.lambdas[i] + (1 - rho) * alpha * init_lambdas_hat[i] + (1 - alpha) * lambdas_hat[i]) for i in range(len(losses))]
        self.lambdas = [var.assign(tf.stop_gradient(lam)) for var, lam in zip(self.lambdas, new_lambdas)]

        # compute weighted loss
        loss = tf.reduce_sum([lam * loss for lam, loss in zip(self.lambdas, losses)])

        # store current losses in self.last_losses to be accessed in the next iteration
        self.last_losses = [var.assign(tf.stop_gradient(loss)) for var, loss in zip(self.last_losses, losses)]
        # in first iteration, store losses in self.init_losses to be accessed in next iterations
        first_iteration = tf.cast(self.call_count < 1, dtype=tf.float32)
        self.init_losses = [var.assign(tf.stop_gradient(loss * first_iteration + var * (1 - first_iteration))) for var, loss in zip(self.init_losses, losses)]
        
        self.call_count.assign_add(1)

        return loss
