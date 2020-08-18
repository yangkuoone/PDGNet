import json

# generate 1 order genes based on PPI data for negative sample screening
class PpiNeighbor(object):
    def __init__(self):
        self.gene_dict = dict()
        self._get_1order_nei_main()

    def _get_1order_nei_main(self):
        self.ppi_file = '../data/ppi_data.txt'
        self.cv_i = '1'    #'0', '1', '2', ..., '9'
        self.cv_file = '../data/dis_gene_cv10.json'
        self.ppi_1nd_file = '../data/ppi_1nd_cv9.txt'
        self.get_1order_nei()

    def get_1order_nei(self):

        # load ppi data
        self.load_ppi()
        # load disease-gene associations
        dis_dict = self.load_dis_gene()

        fw = open(self.ppi_1nd_file, 'w')
        fw.truncate()
        for dis, gset in dis_dict.items():
            nei_dict = dict()
            for g in gset:
                nei_genes = self.gene_dict.get(g)
                if nei_genes is not None:
                    for temp_g in nei_genes:
                        if temp_g not in gset:
                            nei_dict.setdefault(temp_g, 0)
                            nei_dict[temp_g] += 1
            for a, b in nei_dict.items():
                fw.write('\t'.join([dis, a, str(b)])+'\n')

        fw.flush()
        fw.close()

    # load disease-gene associations.
    def load_dis_gene(self):
        train_dis_dict = dict()  # key: train dis_name, value: all the train genes of the dis
        with open(self.cv_file, 'r') as fr:
            json_dict = json.load(fr)
        json_data = json_dict[self.cv_i]
        for dis, gene in json_data['d_train']:
            train_dis_dict.setdefault(dis, set())
            train_dis_dict[dis].add(gene)

        return train_dis_dict

    def load_ppi(self):
        with open(self.ppi_file, 'r') as fr:
            for line in fr:
                g1, g2, _ = line.strip().split('\t')
                self.gene_dict.setdefault(g1, set())
                self.gene_dict.setdefault(g2, set())
                self.gene_dict[g1].add(g2)
                self.gene_dict[g2].add(g1)


if __name__ == '__main__':
    ppi_nei = PpiNeighbor()