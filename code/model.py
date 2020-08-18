# -*- Encoding:UTF-8 -*-
import tensorflow as tf
from layers import Dense

class Model(object):
    def __init__(self, args, data):

        self._add_placeholders()
        self._add_model(args, data)
        self._add_loss(args)
        self._add_train(args)

    def _add_placeholders(self):
        self.dis = tf.placeholder(tf.int32)
        self.gene = tf.placeholder(tf.int32)
        self.dg_label = tf.placeholder(tf.float32)

        # for computing l2 loss
        self.vars_dis = []
        self.vars_gene = []

    def _add_model(self, args, data):

        dis_matrix1 = data[0]
        gene_matrix1 = data[1]
        n_dis_dim = len(dis_matrix1[0])
        n_gene_dim = len(gene_matrix1[0])

        dis_symp_matrix = data[4]
        gene_go_matrix = data[5]
        n_symp = len(dis_symp_matrix[0])
        n_go = len(gene_go_matrix[0])

        self.dis_matrix = tf.convert_to_tensor(dis_matrix1)
        self.gene_matrix = tf.convert_to_tensor(gene_matrix1)
        self.dis_matrix_s = tf.convert_to_tensor(dis_symp_matrix)
        self.gene_matrix_s = tf.convert_to_tensor(gene_go_matrix)

        self.dis_embedding = tf.nn.embedding_lookup(self.dis_matrix, self.dis)
        self.gene_embedding = tf.nn.embedding_lookup(self.gene_matrix, self.gene)
        self.dis_embedding_s = tf.nn.embedding_lookup(self.dis_matrix_s, self.dis)
        self.gene_embedding_s = tf.nn.embedding_lookup(self.gene_matrix_s, self.dis)

        # disease vector based on genes
        with tf.name_scope("Dis_layer"):
            dis_w1 = self.init_variable([n_dis_dim, args.disLayer[0]], "dis_w1")
            self.dis_embedding = tf.matmul(self.dis_embedding, dis_w1)
            for i in range(len(args.disLayer) - 1):
                dis_mlp = Dense(input_dim=args.disLayer[i], output_dim=args.disLayer[i+1])
                self.dis_embedding = dis_mlp(self.dis_embedding)
                self.vars_dis.extend(dis_mlp.vars)

        # disease vector based on symptoms
        with tf.name_scope("Dis_symp_layer"):
            dis_w1_s = self.init_variable([n_symp, args.disLayer_s[0]], "dis_w1_s")
            self.dis_embedding_s = tf.matmul(self.dis_embedding_s, dis_w1_s)
            for i in range(len(args.disLayer_s) - 1):
                dis_mlp_s = Dense(input_dim=args.disLayer_s[i], output_dim=args.disLayer_s[i+1])
                self.dis_embedding_s = dis_mlp_s(self.dis_embedding_s)
                self.vars_dis.extend(dis_mlp_s.vars)

        # gene vector based on diseases
        with tf.name_scope("Gene_layer"):

            gene_w1 = self.init_variable([n_gene_dim, args.geneLayer[0]], "gene_w1")
            self.gene_embedding = tf.matmul(self.gene_embedding, gene_w1)
            for i in range(len(args.geneLayer) - 1):
                gene_mlp = Dense(input_dim=args.geneLayer[i], output_dim=args.geneLayer[i+1])
                self.gene_embedding = gene_mlp(self.gene_embedding)
                self.vars_gene.extend(gene_mlp.vars)

        # gene vector based on go
        with tf.name_scope("Gene_go_layer"):
            gene_w1_s = self.init_variable([n_go, args.geneLayer_s[0]], "gene_w1_s")
            self.gene_embedding_s = tf.matmul(self.gene_embedding_s, gene_w1_s)
            for i in range(len(args.geneLayer_s) - 1):
                gene_mlp_s = Dense(input_dim=args.geneLayer_s[i], output_dim=args.geneLayer_s[i+1])
                self.gene_embedding_s = gene_mlp_s(self.gene_embedding_s)
                self.vars_gene.extend(gene_mlp_s.vars)

        # concat dis_embedding and dis_embedding_s
        self.dis_embedding = tf.concat([self.dis_embedding, self.dis_embedding_s], axis=1)

        # concat gene_embedding and gene_embedding_s
        self.gene_embedding = tf.concat([self.gene_embedding, self.gene_embedding_s], axis=1)

        norm_dis_output = tf.sqrt(tf.reduce_sum(tf.square(self.dis_embedding), axis=1))
        norm_gene_output = tf.sqrt(tf.reduce_sum(tf.square(self.gene_embedding), axis=1))

        # y of dis-gene
        multiply_dg = tf.reduce_sum(tf.multiply(self.dis_embedding, self.gene_embedding), axis=1, keep_dims=False)
        self.dg_y = multiply_dg / (norm_dis_output * norm_gene_output)
        self.dg_y = tf.maximum(1e-6, self.dg_y)

    @staticmethod
    def init_variable(shape, name):
        input_dim, output_dim = shape
        return tf.get_variable(
            name=name,
            shape=(input_dim, output_dim),
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            dtype=tf.float32)

    def _add_loss(self, args):
        # dis-gene loss
        temp = self.dg_label * tf.log(self.dg_y) + (1 - self.dg_label) * tf.log(1 - self.dg_y + 1e-6)
        self.dg_loss = -tf.reduce_sum(temp)
        self.dg_l2_loss = tf.nn.l2_loss(self.dis_embedding) + tf.nn.l2_loss(self.gene_embedding)
        for var in self.vars_dis:
            self.dg_l2_loss += tf.nn.l2_loss(var)
        for var in self.vars_gene:
            self.dg_l2_loss += tf.nn.l2_loss(var)
        # self.dg_loss = self.dg_loss + self.dg_l2_loss * args.l2_weight
        self.dg_loss = self.dg_loss


    def _add_train(self, args):
        self.optimizer_dg = tf.train.AdamOptimizer(args.lr).minimize(self.dg_loss)

    # train dis-gene network
    def train_dg_net(self, sess, feed_dict):
        return sess.run([self.optimizer_dg, self.dg_loss, self.dg_y], feed_dict)

    # predict dis-gene associations
    def predict_dg(self, sess, feed_dict):
        return sess.run([self.dg_y], feed_dict)



