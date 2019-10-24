# -*- Encoding:UTF-8 -*-
class Evaluation(object):
    def __init__(self):
        pass

    # get top predicted genes
    @staticmethod
    def get_top_genes(gene_list, score_list, label_list):
        n_gene = len(gene_list)
        score_gene = []
        hit_list = []
        gene_label = {}  # key: gene, value: rating
        for i in range(n_gene):
            score_gene.append([score_list[i], gene_list[i]])
            gene_label[gene_list[i]] = label_list[i]
        score_gene.sort(reverse=True)
        for score, gene in score_gene[0:1000]:
            if gene_label[gene] == 1.0:
                hit_list.append(1)
            else:
                hit_list.append(0)

        n_known_genes = int(sum(label_list))
        n_topk_hit = sum(hit_list[:n_known_genes])
        return hit_list, n_known_genes, n_topk_hit

    def cal_prf(self, all_hit_list, n_total_test):
        top_k_list = [3, 5, 10]
        all_hit_num_list = []
        test_dis_num = len(all_hit_list)
        prf_summary = []
        for i in range(test_dis_num):
            hit_list = all_hit_list[i]
            hit_num_list = []
            for k in top_k_list:
                hit_num_list.append(sum(hit_list[:k]))
            all_hit_num_list.append(hit_num_list)

        for i in range(len(top_k_list)):
            top_k = top_k_list[i]
            temp = [x[i] for x in all_hit_num_list]
            hit_sum = sum(temp)
            precision = (hit_sum*1.0) / (top_k * test_dis_num)
            recall = (hit_sum*1.0) / n_total_test
            f1 = self.cal_f1(precision, recall)
            prf_summary.append(precision)
            prf_summary.append(recall)
            prf_summary.append(f1)

        return prf_summary


    def cal_metrics(self, gene_array, score_array, rating):
        n_gene = len(gene_array)
        # print('n_gene_array:', n_gene)
        # print('n_score_array:', len(score_array))
        score_gene = []
        hit_list = []
        gene_rating = {}  # key: gene, value: rating
        for i in range(n_gene):
            score_gene.append([score_array[i], gene_array[i]])
            gene_rating[gene_array[i]] = rating[i]
        score_gene.sort(reverse=True)
        for score, gene in score_gene:
            if gene_rating[gene] == 1.0:
                hit_list.append(1)
            else:
                hit_list.append(0)
        known_gene_num = int(sum(rating))
        prf = self.cal_pr_re_f1(hit_list, known_gene_num)  # prf means precision, recall, f1-measure
        topk_hit = sum(hit_list[:known_gene_num])

        return prf, topk_hit

    @staticmethod
    def cal_prf_avg(prf_array):
        prf_avg = list()
        array_num = len(prf_array)
        for i in range(len(prf_array[0])):
            temp = sum(prf_array[:, i]) / array_num
            prf_avg.append(temp)
        return prf_avg


    # calculate pr, re, f1
    def cal_pr_re_f1(self, hit_list, known_gene_num):
        top_k_list = [3, 5, 10]
        prf_list = []
        for k in top_k_list:
            topk_pr = sum(hit_list[:k]) / float(k)
            topk_re = sum(hit_list[:k]) / float(known_gene_num)
            topk_f1 = self.cal_f1(topk_pr, topk_re)
            prf_list.append(topk_pr)
            prf_list.append(topk_re)
            prf_list.append(topk_f1)
        return prf_list

    # calculate F-measure
    @staticmethod
    def cal_f1(precision, recall):
        if precision == 0 or recall == 0: return 0
        return precision * recall * 2 / (precision + recall)
