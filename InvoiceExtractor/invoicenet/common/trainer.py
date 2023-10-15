# Copyright (c) 2019 Tradeshift
# Copyright (c) 2020 Sarthak Mittal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from invoicenet.acp.acp import AttendCopyParse

from invoicenet.common.model import Model


def train(model: Model,
          train_data: tf.data.Dataset,
          val_data: tf.data.Dataset,
          total_steps=5000,
          early_stop_steps=0):

    print_interval = 20
    no_improvement_steps = 0
    best = float("inf")

    train_iter = iter(train_data)
    val_iter = iter(val_data)

    start = time.time()

    list_train_loss = []
    list_val_loss = []
    list_steps = []

    list_val_acc = []
    list_train_acc = []

    for step in range(total_steps):
        try:
            train_loss = model.train_step(next(train_iter))
            train_acc = AttendCopyParse.train_acc_metric.result()

        except StopIteration:
            print(
                "Couldn't find any training data! Have you prepared your training data?")
            print("Terminating...")
            break

        if not np.isfinite(train_loss):
            raise ValueError("NaN loss")

        if step % print_interval == 0:
            took = time.time() - start

            try:
                val_loss = model.val_step(next(val_iter))
                val_acc = AttendCopyParse.val_acc_metric.result()

            except StopIteration:
                print(
                    "Couldn't find any validation data! Have you prepared your training data?")
                print("Terminating...")
                break

            print("[%d/%d | %.2f steps/s]: train acc: %.4f val acc: %.4f train loss: %.4f val loss: %.4f" % (
                step, total_steps, (step + 1) / took, train_acc, val_acc, train_loss, val_loss))

            list_train_loss.append(float(train_loss))
            list_val_loss.append(float(val_loss))
            list_train_acc.append(float(train_acc))
            list_val_acc.append(float(val_acc))
            list_steps.append(float(step))

            if not np.isfinite(val_loss):
                raise ValueError("NaN loss")
            if val_loss < best:
                no_improvement_steps = 0
                best = val_loss
                model.save("best")
            elif early_stop_steps > 0:
                no_improvement_steps += print_interval
                if no_improvement_steps >= early_stop_steps:
                    print("Validation loss has not improved for {} steps, terminating!".format(
                        no_improvement_steps))
                    return

    plt.plot(list_steps, list_train_loss)
    plt.plot(list_steps, list_val_loss)
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(list_steps, list_train_acc)
    plt.plot(list_steps, list_val_acc)
    plt.title('Model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
