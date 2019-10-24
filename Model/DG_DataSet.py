# -*- Encoding:UTF-8 -*-

import numpy as np
import json
import random


class DataSet(object):
    def __init__(self):
        self.dis_NI = {}
        self.dis_IN = {}
        self.gene_NI = {}
        self.gene_IN = {}

        self.cv_file = '../../data/dis_gene_cv10.json'
        self.dis_symp_file = '../../data/hpo&orpha_dis_symp.txt'
        self.gene_go_file = '../../data/homo_gene_GO.txt'
        self.dis_sim_file = '../../data/cv10_of0_dis_sim_cos.txt'
        self.gene_sim_file = '../../data/cv10_of0_gene_sim_cos.txt'
        self.cf_gene_file = '../../data/cf_candidate_cv0.txt'
        self.ppi_gene_file = '../../data/ppi_1nd_cv0.txt'

        self.cv_i = '0'

        self.cv_data, self.n_dis, self.n_gene = self.load_cv_file()

        self.candidate_gene_set = self.load_pre_gene()

        # build train data
        # self.train_data = self.build_train_data()
        # build test data
        self.test_data = self.build_test_data()

        self.dis_matrix, self.gene_matrix = self.build_dis_gene_matrix()

        self.dis_symp_matrix = self.load_dis_symp()
        self.gene_go_matrix = self.load_gene_go()

        self.train_dis_sim_data = self.load_dis_sim()
        self.train_gene_sim_data = self.load_gene_sim()

        self.print_data_info()

    def print_data_info(self):
        print('loading data success...')
        print('n_disease:', self.n_dis)
        print('n_gene:', self.n_gene)
        # print('n_train_data:', len(self.train_data[0]))
        print('n_test_data:', len(self.test_data[0]))
        print('n_train_dis_sim:', len(self.train_dis_sim_data[0]))
        print('n_train_gene_sim:', len(self.train_gene_sim_data[0]))
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

            temp_set = train_gene_set - gset
            n_samples = min([int(len(temp_set) / 2), len(gset) * n_neg_samples])
            rand_gset = random.sample(temp_set, n_samples)
            for g in rand_gset:
                dis_list.append(d)
                gene_list.append(g)
                label_list.append(0.0)
        return np.array(dis_list), np.array(gene_list), np.array(label_list)


    def build_test_data(self):
        print('build test data ...')
        dis_list = []
        gene_list = []
        label_list = []
        test_dis_dic = self.cv_data.get('test_dis_dic')
        train_gene = self.cv_data.get('train_gene')
        train_dis_dict = self.cv_data.get('train_dis_dic')
        all_dis_list = list(self.dis_NI.keys())
        all_dis_list.sort()
        selected_dis_list = all_dis_list[0:200]
        for d, gset in test_dis_dic.items():
            # if self.dis_IN.get(d) not in selected_dis_list: continue
            temp_dis = []
            temp_gene = []
            temp_rating = []
            for g in gset:
                temp_dis.append(d)
                temp_gene.append(g)
                temp_rating.append(1.0)
            train_gset = train_dis_dict.get(d)
            for g in train_gene:
                if (d, g) not in self.candidate_gene_set: continue
                if g not in gset and g not in train_gset:
                    temp_dis.append(d)
                    temp_gene.append(g)
                    temp_rating.append(0.0)
            dis_list.append(temp_dis)
            gene_list.append(temp_gene)
            label_list.append(temp_rating)

        return [np.array(dis_list), np.array(gene_list), np.array(label_list)]


    # load disease-symptom associations
    def load_dis_symp(self):

        symp_dis_dict = dict()
        with open(self.dis_symp_file, 'r') as fr:
            for line in fr:
                dis, symp = line.strip().split('\t')
                symp_dis_dict.setdefault(symp, set())
                symp_dis_dict[symp].add(dis)

        symp_NI = {}
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


    # load gene-GO data
    def load_gene_go(self):
        go_gene_dict = dict()
        with open(self.gene_go_file, 'r') as fr:
            for line in fr:
                gene, go = line.strip().split('\t')
                go_gene_dict.setdefault(go, set())
                go_gene_dict[go].add(gene)

        counter = 0
        go_NI = {}
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

    def load_dis_sim(self):

        dis1_list = list()
        dis2_list = list()
        label_list = list()
        with open(self.dis_sim_file, 'r') as fr:
            for line in fr:
                dis1, dis2, score = line.strip().split('\t')
                did1 = self.dis_NI.get(dis1)
                did2 = self.dis_NI.get(dis2)
                if did1 is not None and did2 is not None:
                    dis1_list.append(did1)
                    dis2_list.append(did2)
                    label_list.append(float(score))
        return np.array(dis1_list), np.array(dis2_list), np.array(label_list)

    def load_gene_sim(self):
        gene1_list = list()
        gene2_list = list()
        label_list = list()
        with open(self.gene_sim_file, 'r') as fr:
            for line in fr:
                gene1, gene2, score = line.strip().split('\t')
                gid1 = self.gene_NI.get(gene1)
                gid2 = self.gene_NI.get(gene2)
                if gid1 is not None and gid2 is not None:
                    gene1_list.append(gid1)
                    gene2_list.append(gid2)
                    label_list.append(float(score))
        return np.array(gene1_list), np.array(gene2_list), np.array(label_list)

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


    def load_pre_gene(self):

        candidate_gene_set = set()
        with open(self.cf_gene_file, 'r') as fr:
            for line in fr:
                dis, gene, score = line.strip().split('\t')
                did = self.dis_NI.get(dis)
                gid = self.gene_NI.get(gene)
                if did is None or gid is None: continue
                candidate_gene_set.add((did, gid))

        with open(self.ppi_gene_file, 'r') as fr:
            for line in fr:
                dis, gene, score = line.strip().split('\t')
                did = self.dis_NI.get(dis)
                gid = self.gene_NI.get(gene)
                if did is None or gid is None: continue
                candidate_gene_set.add((did, gid))

        return candidate_gene_set

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
