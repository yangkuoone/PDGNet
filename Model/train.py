import tensorflow as tf
import numpy as np
import argparse
from DG_DataSet import DataSet
from Evaluation import Evaluation
import sys
import os
from model import Model

class Train(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self._set_args()
        self.args.dim = 6144
        self.args.disLayer = [6144, 6144]
        self.args.geneLayer = [6144, 6144]
        self.args.disLayer_s = [4096, 4096]
        self.args.geneLayer_s = [4096, 4096]
        self.args.maxEpochs = 100
        self.args.negNum = 50
        self.args.l2_weight = 1e-5
        self.data_set = DataSet()
        self.train()


    def _set_args(self):
        parser = argparse.ArgumentParser(description="Options")
        parser.add_argument('-negNum', action='store', dest='negNum', default=70, type=int)
        parser.add_argument('-dim', action='store', dest='dim', default=1024)
        parser.add_argument('-disLayer', action='store', dest='disLayer', default=[8192, 8192])
        parser.add_argument('-geneLayer', action='store', dest='geneLayer', default=[8192, 8192])
        parser.add_argument('-disLayer_s', action='store', dest='disLayer_s', default=[8192, 8192])
        parser.add_argument('-geneLayer_s', action='store', dest='geneLayer_s', default=[8192, 8192])
        parser.add_argument('-lr', action='store', dest='lr', default=0.000001)
        parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)
        parser.add_argument('-batchSize', action='store', dest='batchSize', default=128, type=int)
        parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=20)
        parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
        parser.add_argument('-topK', action='store', dest='topK', default=10)
        parser.add_argument('-testDisNum', action='store', dest='nTestDis', default=200)
        parser.add_argument('-l2_weight', action='store', dest='l2_weight', default=1e-5)
        parser.add_argument('-simPlus', action='store', dest='simPlus', default=0.2)
        parser.add_argument('-train_interval', action='store', dest='train_interval', default=2)

        self.args = parser.parse_args()

        # setting gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True


    def train(self):
        print('train model...')
        dis_matrix = self.data_set.dis_matrix
        gene_matrix = self.data_set.gene_matrix
        n_dis = self.data_set.n_dis
        n_gene = self.data_set.n_gene

        data = [dis_matrix, gene_matrix, n_dis, n_gene,
                self.data_set.dis_symp_matrix,
                self.data_set.gene_go_matrix]
        model = Model(self.args, data)
        best_ap = 0
        best_epoch = -1
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.args.maxEpochs):
                print("=" * 20 + "Epoch ", epoch, "=" * 20)
                self.run_epoch(sess, model, epoch)
                print('=' * 50)
                print("Start Evaluation!")

                # prf_avg, avg_pr = self.evaluate(sess, model)
                prf_summary, ap, n_total_hit, n_total_test = self.evaluate(sess, model)
                top3_pr = prf_summary[0]
                top10_pr = prf_summary[6]
                print 'epoch:', epoch, '; AP:', ap, '; top@3 pr:', top3_pr, '; top@10 pr:', top10_pr
                result = [n_total_test, n_total_hit, ap] + prf_summary
                result = [str(x) for x in result]
                if best_ap < ap:
                    best_ap = ap
                    best_epoch = epoch

                print('\t'.join(result))
                if epoch - best_epoch > self.args.earlyStop:    # early stop
                    print("Normal Early stop!")
                    break
                print("=" * 20 + "Epoch ", epoch, "End" + "=" * 20)
            print("Training complete!")


    def run_epoch(self, sess, model, epoch):
        # train dis-gene network
        self.train_dg_net(sess, model)


    def train_dg_net(self, sess, model, verbose=1000):

        # train_dis, train_gene, label = self.data_set.train_data
        train_dis, train_gene, label = self.data_set.build_train_data(self.args.negNum)

        train_len = len(train_dis)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_dis = train_dis[shuffled_idx]
        train_gene = train_gene[shuffled_idx]
        label = label[shuffled_idx]

        num_batches = train_len // self.args.batchSize + 1

        losses = []
        for i in range(num_batches):
            # if i > 1000: break
            min_idx = i * self.args.batchSize
            max_idx = np.min([train_len, (i + 1) * self.args.batchSize])
            train_d_batch = train_dis[min_idx: max_idx]
            train_g_batch = train_gene[min_idx: max_idx]
            train_l_batch = label[min_idx: max_idx]

            feed_dict = {model.dis: train_d_batch,
                         model.gene: train_g_batch,
                         model.dg_label: train_l_batch}

            _, loss, y = model.train_dg_net(sess, feed_dict)
            losses.append(loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in DG net is: {}".format(loss))
        return loss

    def evaluate(self, sess, model):
        evaluation = Evaluation()
        test_dis, test_gene, test_label = self.data_set.test_data

        # prf_list = []  # store precision, recall, f1-measure
        n_total_hit = 0.0
        n_total_test = 0.0
        dis_num = len(test_dis)
        all_hit_list = list()
        for i in range(dis_num):
            if i % 100 == 0:
                print i, '/', dis_num
            feed_dict = {model.dis: test_dis[i],
                         model.gene: test_gene[i]}
            predict = model.predict_dg(sess, feed_dict)
            # prf, topk_hit = evaluation.cal_metrics(test_gene[i], predict[0], test_label[i])
            hit_list, n_known_genes, n_topk_hit = evaluation.get_top_genes(test_gene[i], predict[0], test_label[i])
            n_total_hit += n_topk_hit
            n_total_test += n_known_genes
            # total_edges_num += sum(test_label[i])
            # prf_list.append(prf)
            all_hit_list.append(hit_list)

        ap = n_total_hit / n_total_test

        prf_summary = evaluation.cal_prf(all_hit_list, n_total_test)
        # prf_array = np.array(prf_list)
        # prf_avg = evaluation.cal_prf_avg(prf_array)
        return prf_summary, ap, n_total_hit, n_total_test


if __name__ == '__main__':
    t = Train()
