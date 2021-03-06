import tensorflow as tf
import tqdm
from few_shot_network import FewshotsNet
import os
import glob
import numpy as np
from collections import defaultdict

path = os.getcwd()


class ExperimentBuilder:

    def __init__(self, data):
        """
        Initializes an ExperimentBuilder object. The ExperimentBuilder object takes care of setting up our experiment
        and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
        and evaluation procedures.
        :param data: A data provider class
        """
        self.data = data


    # def whydidyounotjoinusformoredrinks(self):
    #
    #     print "we went for drinks with Ryutaor"





    def build_experiment(self, batch_size, ways, shots, query_size, image_shape, data_format):

        """

        :param batch_size: The experiment batch size
        :param ways: An integer indicating the number of classes per support set
        :param shots: An integer indicating the number of samples per class
        :param channels: The image channels
        :param fce: Whether to use full context embeddings or not
        :return: a few_shot_learning object, along with the losses, the training ops and the init op
        """
        height, width, channels = image_shape
        self.support_set_x = tf.placeholder(tf.float32, shape = [batch_size, ways, shots, height, width,
                                            channels], name = 'support_set_images')
        self.support_set_y = tf.placeholder(tf.uint8, shape = [batch_size, ways, shots], name = 'support_set_labels')
        self.query_x = tf.placeholder(tf.float32, shape = [batch_size, query_size*ways, height, width, channels], name = 'query_images')
        self.query_y = tf.placeholder(tf.uint8, shape = [batch_size, query_size*ways], name = 'query_labels')
        self.is_training = tf.placeholder(tf.bool, name='training_flag')
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        # self.support_set_x_cand = tf.placeholder(tf.float32, shape=[20,ways,shots,height,width,channels], name="support_set_images_cand")
        # self.support_set_y_cand = tf.placeholder(tf.uint8,shape=[20,ways,shots], name="support_set_labels_cand")
        # self.query_x_cand = tf.placeholder(tf.float32,shape=[20,1,height,width,channels],name="query_images_cand")
        # self.query_y_cand =tf.placeholder(tf.uint8,shape=[20,1], name="query_labels_cand")
        self.current_learning_rate = 0.0001
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.few_shot_miniImagenet = FewshotsNet(batch_size=batch_size, support_set_images=self.support_set_x,
                                                 support_set_labels=self.support_set_y,
                                                 query_images=self.query_x, query_labels=self.query_y, data_format = data_format,
                                                 dropout_prob=self.dropout_prob,
                                                 is_training=self.is_training, learning_rate=self.learning_rate)

        self.losses, self.ada_opts = self.few_shot_miniImagenet.init_train()
        init = tf.global_variables_initializer()
        self.total_train_iter = 0
        self.total_test_iter = 0
        self.total_val_iter = 0
        return self.few_shot_miniImagenet, self.losses, self.ada_opts, init

    def run_training_epoch(self, total_training_episodes, writer, sess):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        losses = []
        accuracies = []
        with tqdm.tqdm(total=total_training_episodes) as pbar:

            for i in range(total_training_episodes):  # train epoch
                support_set_x, support_set_y, query_x, query_y = self.data.get_tra_batch()
                _, c_loss_value, acc = sess.run(
                    [self.ada_opts, self.losses['loss'], self.losses['accuracy']],
                    feed_dict={self.dropout_prob: 0.5, self.support_set_x: support_set_x,
                               self.support_set_y: support_set_y, self.query_x: query_x, self.query_y: query_y,
                               self.is_training: True, self.learning_rate: self.current_learning_rate})


                tf.logging.info('train_loss:{}, accuracy:{}'.format(c_loss_value, acc))
                self.total_train_iter += 1
                pbar.update(1)
                losses.append(c_loss_value)
                accuracies.append(acc)
                if self.total_train_iter % 100 == 0:
                    train_summary = tf.Summary()
                    loss_last_100 = np.mean(losses[-100:])

                    accuracy_last_100 = np.mean(accuracies[-100:])

                    train_summary.value.add(tag='loss', simple_value=loss_last_100)
                    train_summary.value.add(tag='accuracy', simple_value = accuracy_last_100)
                    writer.add_summary(train_summary, self.total_train_iter)
                if self.total_train_iter % 2000 == 0:
                    self.current_learning_rate /= 2
                    print("change learning rate", self.current_learning_rate)

        average_accuracy = np.mean(accuracies)
        average_loss = np.mean(losses)
        return average_loss, average_accuracy

    def run_validation_epoch(self, total_val_episodes, writer, sess):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        losses = []
        accuracies = []

        with tqdm.tqdm(total=total_val_episodes) as pbar:
            for i in range(total_val_episodes):  # validation epoch
                support_set_x, support_set_y, query_x, query_y = self.data.get_val_test_batch("val")

                c_loss_value, acc = sess.run(
                    [self.losses['loss'], self.losses['accuracy']],
                    feed_dict={self.dropout_prob: 1.0, self.support_set_x: support_set_x,
                               self.support_set_y: support_set_y, self.query_x: query_x, self.query_y: query_y,
                               self.is_training: False})

                tf.logging.info('val_loss:{}, accuracy:{}'.format(c_loss_value, acc))
                pbar.update(1)

                losses.append(c_loss_value)
                accuracies.append(acc)
                self.total_val_iter += 1

            mean_loss = np.mean(losses)
            mean_accuracy = np.mean(accuracies)
            val_summary = tf.Summary()
            val_summary.value.add(tag='loss', simple_value=mean_loss)
            val_summary.value.add(tag='accuracy', simple_value=mean_accuracy)
            writer.add_summary(val_summary, self.total_val_iter)

        return mean_loss, mean_accuracy

    def run_testing_epoch(self, total_test_episodes, sess, writer):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        losses = []
        accuracies = []
        with tqdm.tqdm(total=total_test_episodes) as pbar:
            for i in range(total_test_episodes):
                support_set_x, support_set_y, query_x, query_y = self.data.get_val_test_batch("test")
                c_loss_value, acc = sess.run(
                    [self.losses['loss'], self.losses['accuracy']],
                    feed_dict={self.dropout_prob: 1.0, self.support_set_x: support_set_x,
                               self.support_set_y: support_set_y, self.query_x: query_x,
                               self.query_y: query_y,
                               self.is_training: False})

                tf.logging.info('test_loss:{}, accuracy:{}'.format(c_loss_value, acc))
                losses.append(c_loss_value)
                accuracies.append(acc)
                pbar.update(1)
                self.total_test_iter += 1

                
            test_summary = tf.Summary()
            mean_loss = np.mean(losses)
            mean_accuracy = np.mean(accuracies)
            test_summary.value.add(tag='loss', simple_value=mean_loss)
            test_summary.value.add(tag='accuracy', simple_value=mean_accuracy)
            writer.add_summary(test_summary, self.total_train_iter)
    
        return mean_loss, mean_accuracy

    def run_find_cand_epoch(self,sess):
        """
        validate the current classifier on candidate training data
        to find the most uncertain several images for each class
        """
        cand_imgs_dir = path + '/miniImagenet/tra_cand_data_mini/'
        # class_ = glob.glob(cand_imgs_dir + 'n*')
        total_episode = 0
        selected_imgs = defaultdict(list)
        for i in range(5):
            imgs_dir = cand_imgs_dir + str(i)
            imgs = glob.glob(imgs_dir + '/*.png')
            uncertain_dic = defaultdict(list)
            top_50_unceratin_imgs = []
            for j, img in enumerate(imgs):

                support_set_x, support_set_y, query_x, query_y = self.data.get_cand_data(i, img)
                uncertainty = sess.run([self.losses['loss']],
                                       feed_dict={self.dropout_prob: 1.0,
                                                  self.support_set_x:support_set_x,
                                                  self.support_set_y:support_set_y,
                                                  self.query_x:query_x,
                                                  self.query_y:query_y,
                                                  self.is_training:False})

                uncertain_dic[img].append(uncertainty)
                total_episode += 1
                print(uncertainty, img)
            uncertain_dic = sorted(uncertain_dic.items(), key= lambda x: x[1], reverse=True)
            for img_ in uncertain_dic[0:20]:
                img_name = img_[0]
                top_50_unceratin_imgs.append(img_name)

            selected_imgs[i].append(top_50_unceratin_imgs)
        return selected_imgs


