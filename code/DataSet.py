# -*- Encoding:UTF-8 -*-

import numpy as np
import json
import random

class DataSet(object):
    def __init__(self):
        self.dis_NI = dict()
        self.dis_IN = dict()
        self.gene_NI = dict()
        self.gene_IN = dict()

        self.cv_file = '../data/dis_gene_cv10.json'
        self.dis_symp_file = '../data/hpo&orpha_dis_symp.txt'
        self.gene_go_file = '../data/homo_gene_GO.txt'
        self.ppi_gene_file = '../data/ppi_1nd_cv0.txt'

        self.cv_i = '0'  # index for cross validation

        self.cv_data, self.n_dis, self.n_gene = self.load_cv_file()

        self.candidate_dict = self.load_ppi_data()

        # build test data
        self.test_data = self.build_test_data()

        self.dis_matrix, self.gene_matrix = self.build_dis_gene_matrix()

        self.dis_symp_matrix = self.load_dis_symp_data()
        self.gene_go_matrix = self.load_gene_go_data()

        self.print_data_info()

    def print_data_info(self):
        print('loading data...')
        print('n_disease:', self.n_dis)
        print('n_gene:', self.n_gene)
        print('n_test_data:', len(self.test_data[0]))
        print('n_symptom:', len(self.dis_symp_matrix[0]))
        print('n_go:', len(self.gene_go_matrix[0]))
        print('n_dimension (dis_matrix):', len(self.dis_matrix[0]))
        print('n_dimension (gene_matrix):', len(self.gene_matrix[0]))


    def build_train_data(self, n_neg_samples):
        print('build train data ...')
        dis_list = list()
        gene_list = list()
        label_list = list()

        train_dis_dict = self.cv_data.get('train_dis_dic')
        train_gene_set = self.cv_data.get('train_gene')
        for d, gset in train_dis_dict.items():
            for g in gset:
                dis_list.append(d)
                gene_list.append(g)
                label_list.append(1.0)

            candidate_gene = self.candidate_dict.get(d)
            if candidate_gene is None:
                temp_set = train_gene_set - gset
            else:
                temp_set = train_gene_set - gset - candidate_gene
            n_samples = min([int(len(temp_set) / 2), len(gset) * n_neg_samples])
            rand_gset = random.sample(temp_set, n_samples)
            for g in rand_gset:
                dis_list.append(d)
                gene_list.append(g)
                label_list.append(0.0)
        return np.array(dis_list), np.array(gene_list), np.array(label_list)

    def build_test_data(self):
        print('building test data...')
        dis_list = []
        gene_list = []
        label_list = []
        test_dis_dic = self.cv_data.get('test_dis_dic')
        train_gene = self.cv_data.get('train_gene')
        train_dis_dict = self.cv_data.get('train_dis_dic')
        all_dis_list = list(self.dis_NI.keys())
        all_dis_list.sort()
        for d, gset in test_dis_dic.items():
            temp_dis = []
            temp_gene = []
            temp_rating = []
            for g in gset:
                temp_dis.append(d)
                temp_gene.append(g)
                temp_rating.append(1.0)
            train_gset = train_dis_dict.get(d)

            can_gene = self.candidate_dict.get(d)
            if can_gene is None:
                neg_gene_set = train_gene - gset - train_gset
            else:
                neg_gene_set = train_gene - gset - train_gset - can_gene
            for g in neg_gene_set:
                temp_dis.append(d)
                temp_gene.append(g)
                temp_rating.append(0.0)

            dis_list.append(temp_dis)
            gene_list.append(temp_gene)
            label_list.append(temp_rating)

        return [np.array(dis_list), np.array(gene_list), np.array(label_list)]

    # load disease-symptom associations
    def load_dis_symp_data(self):
        symp_dis_dict = dict()
        with open(self.dis_symp_file, 'r') as fr:
            for line in fr:
                dis, symp = line.strip().split('\t')
                symp_dis_dict.setdefault(symp, set())
                symp_dis_dict[symp].add(dis)
        symp_NI = dict()
        counter = 0
        for symp, dis_set in symp_dis_dict.items():
            if len(dis_set) <= 2:continue
            symp_NI[symp] = counter
            counter += 1

        dis_symp_matrix = np.zeros([self.n_dis, len(symp_NI)], dtype=np.float32)
        for symp, dis_set in symp_dis_dict.items():
            sid = symp_NI.get(symp)
            if sid is None: continue
            for dis in dis_set:
                did = self.dis_NI.get(dis)
                if did is None: continue
                dis_symp_matrix[did][sid] = 1

        return np.array(dis_symp_matrix)

    # load gene-GO associations
    def load_gene_go_data(self):
        go_gene_dict = dict()
        with open(self.gene_go_file, 'r') as fr:
            for line in fr:
                gene, go = line.strip().split('\t')
                go_gene_dict.setdefault(go, set())
                go_gene_dict[go].add(gene)

        counter = 0
        go_NI = dict()
        for go, gene_set in go_gene_dict.items():
            if len(gene_set) <= 3: continue
            go_NI[go] = counter
            counter += 1

        gene_go_matrix = np.zeros([self.n_gene, len(go_NI)], dtype=np.float32)
        for go, gene_set in go_gene_dict.items():
            go_id = go_NI.get(go)
            if go_id is None: continue
            for gene in gene_set:
                gid = self.gene_NI.get(gene)
                if gid is None: continue
                gene_go_matrix[gid][go_id] = 1

        return np.array(gene_go_matrix)

    def build_dis_gene_matrix(self):
        # train_matrix: user_num * item_num dimension
        train_matrix = np.zeros([self.n_dis, self.n_gene], dtype=np.float32)
        train_list = self.cv_data.get('train_list')
        for d, g in train_list:
            train_matrix[d][g] = 1

        # disease matrix
        del_index_list1 = list()
        for i in range(len(train_matrix[0])):
            val_sum = sum(train_matrix[:, i])
            if val_sum <= 1:
                del_index_list1.append(i)
        dis_matrix = np.delete(train_matrix, del_index_list1, axis=1)

        # gene matrix
        del_index_list2 = list()
        train_matrix_t = train_matrix.T
        for i in range(len(train_matrix_t[0])):
            val_sum = sum(train_matrix_t[:, i])
            if val_sum <= 1:
                del_index_list2.append(i)
        gene_matrix = np.delete(train_matrix_t, del_index_list2, axis=1)

        return dis_matrix, gene_matrix


    # load train and test edges and nodes from edge_train_test_file
    def load_cv_file(self):
        print('load cv data ...')
        train_list = []  # disease-gene associations that are used to train
        test_list = []  # disease-gene associations that are used to test
        train_dis_dic = {}  # key: train dis_name, value: all the train genes of the dis
        test_dis_dic = {}  # key: test dis_name, value: all the test genes of the dis
        train_gene_set = set()  # all genes that are used to train
        test_gene_set = set()  # all genes that are used to test
        data = {}
        with open(self.cv_file, 'r') as fr:
            json_dict = json.load(fr)
        json_data = json_dict[self.cv_i]
        dis_set = set()
        gene_set = set()
        for dis, gene in json_data['d_train']:
            dis_set.add(dis)
            gene_set.add(gene)
        self.build_dict(dis_set, gene_set)
        for dis, gene in json_data['d_train']:
            did = self.dis_NI.get(dis)
            gid = self.gene_NI.get(gene)
            train_dis_dic.setdefault(did, set())
            train_dis_dic[did].add(gid)
            train_list.append((did, gid))
            train_gene_set.add(gid)
        for dis, gene in json_data['e_test']:
            did = self.dis_NI.get(dis)
            gid = self.gene_NI.get(gene)
            test_dis_dic.setdefault(did, set())
            test_dis_dic[did].add(gid)
            test_list.append((did, gid))
            test_gene_set.add(gid)
        data['train_list'] = train_list
        data['test_list'] = test_list
        data['train_dis_dic'] = train_dis_dic
        data['test_dis_dic'] = test_dis_dic
        data['train_gene'] = train_gene_set
        data['test_gene'] = test_gene_set
        n_dis = len(train_dis_dic)
        n_gene = len(train_gene_set)

        return data, n_dis, n_gene

    # loading 1 order genes based on PPI data for negative sample screening
    def load_ppi_data(self):
        candidate_dict = dict()
        with open(self.ppi_gene_file, 'r') as fr:
            for line in fr:
                dis, gene, score = line.strip().split('\t')
                did = self.dis_NI.get(dis)
                gid = self.gene_NI.get(gene)
                if did is None or gid is None: continue
                candidate_dict.setdefault(did, set())
                candidate_dict[did].add(gid)

        return candidate_dict

    def build_dict(self, dis_set, gene_set):
        i = 0
        for d in dis_set:
            self.dis_NI[d] = i
            self.dis_IN[i] = d
            i += 1
        j = 0
        for g in gene_set:
            self.gene_NI[g] = j
            self.gene_IN[j] = g
            j += 1
