import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

"""
Data sets selection:
0: Abt-Buy
1: DBLP-Scholar
2: songs
3: Amazon-Google
4: restaurant
5: products
6: DBLP-ACM
7: Citeseer-DBLP
8: Data shifting: Abt-Buy to Amazon-Google
9: Data shifting: DBLP-Scholar to DBLP-ACM (DS2DA)
10: Data shifting: DBLP-ACM to DBLP-Scholar (DA2DS)
11: Walmart-Amazon
12: Data shifting: Abt-Buy to Walmart-Amazon
13: Data shifting: Walmart-Amazon to Abt-Buy
14: Itunes-Amazon
15: Data shifting: songs to Itunes-Amazon (SG2IA)
16: IA-transfer
17: SG-transfer
18: Data shifting: Itunes-Amazon to songs (IA2SG)
19: cora
20: Data shifting: others to DS
21: Data shifting: others to Cora
22: Data shifting: others to DA
23: Data shifting: cora to DS (cora2DS)
24: Data shifting: DA to cora (DA2cora)
25: Data shifting: cora to DA (cora2DA)
26: Data shifting: DS to cora (DS2cora)
"""

global_selection = 0
tvt_selection = 15


class Configuration(object):
    def __init__(self, selection):
        self.data_selection = selection
        self.train_valida_test_ratio = {
            0: "622",
            1: "522",
            2: "1022",
            3: "2022",
            4: "5022",
            9: "All",
            10: "else",
            11: "127",
            12: "25022",
            13: "226",
            14: "424",
            15: "325",
            16: "523",
            17: "1026",
            18: "3026",
            19: "5026",
            20: "7026",
            21: "10146",
            22: "10246",
            23: "10346",
            24: "2146",
            25: "2246",
            26: "2346",
        }
        self.path_dict = {
            0: rootPath + "/input_data_2020/Abt-Buy/",
            1: rootPath + "/input_data_2020/DBLP-Scholar/",
            2: rootPath + "/input_data_2020/songs/",
            3: rootPath + "/input_data_2020/Amazon-Google/",
            4: rootPath + "/input_data_2020/restaurant/",
            5: rootPath + "/input_data_2020/products/",
            6: rootPath + "/input_data_2020/DBLP-ACM/",
            7: rootPath + "/input_data_2020/Citeseer-DBLP/",
            8: rootPath + "/input_data_2020/AB2AG/",
            9: rootPath + "/input_data_2020/DS2DA/",
            10: rootPath + "/input_data_2020/DA2DS/",
            11: rootPath + "/input_data_2020/Walmart-Amazon/",
            12: rootPath + "/input_data_2020/AB2WA/",
            13: rootPath + "/input_data_2020/WA2AB/",
            14: rootPath + "/input_data_2020/Itunes-Amazon/",
            15: rootPath + "/input_data_2020/SG2IA/",
            16: rootPath + "/input_data_2020/IA-transfer/",
            17: rootPath + "/input_data_2020/SG-transfer/",
            18: rootPath + "/input_data_2020/IA2SG/",
            19: rootPath + "/input_data_2020/cora/",
            20: rootPath + "/input_data_2020/2DS/",
            21: rootPath + "/input_data_2020/2cora/",
            22: rootPath + "/input_data_2020/2DA/",
            23: rootPath + "/input_data_2020/cora2DS/",
            24: rootPath + "/input_data_2020/DA2cora/",
            25: rootPath + "/input_data_2020/cora2DA/",
            26: rootPath + "/input_data_2020/DS2cora/",
        }

        self.tvt_selection = tvt_selection
        # if self.data_selection == 2:
        #     self.tvt_selection = 4
        self.risk_training_size = 20
        self.random_select_risk_training = False
        self.risk_confidence = 0.9
        self.minimum_observation_num = 5
        self.budget_levels = [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
            2500,
            3000,
            3500,
            4000,
            4500,
            5000,
        ]
        self.interval_number_4_continuous_value = 50
        self.learn_variance = True
        self.apply_function_to_weight_classifier_output = True
        self.test_risk_learning_rate = 1e-4
        self.deepmatcher_epochs = 20
        self.risk_training_epochs = 500
        self.use_selected_validation_set = False
        self.only_test_risk_loss = True

    def get_parent_path(self):
        return (
            self.path_dict.get(self.data_selection)
            + self.train_valida_test_ratio.get(self.tvt_selection)
            + "/"
        )

    def get_data_source_1(self):
        # return self.data_source1_dict.get(self.data_selection)
        return self.path_dict.get(self.data_selection) + "tableA.csv"

    def get_data_source_2(self):
        # return self.data_source2_dict.get(self.data_selection)
        if self.data_selection == 2 or self.data_selection == 19:
            return None
        return self.path_dict.get(self.data_selection) + "tableB.csv"

    def get_raw_data_path(self):
        return self.path_dict.get(self.data_selection) + "pair_info_more.csv"

    def get_shift_raw_data_path(self):
        return self.path_dict.get(self.data_selection) + "pair_info_more_2.csv"

    def get_raw_decision_tree_rules_path(self):
        return self.get_parent_path() + "decision_tree_rules_raw.txt"

    def get_decision_tree_rules_path(self):
        return self.get_parent_path() + "decision_tree_rules_clean.txt"

    def use_other_domain_workload(self):
        return self.data_selection in {
            8,
            9,
            10,
            12,
            13,
            15,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
        }

    @staticmethod
    def get_budgets():
        budget_levels = [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
            2500,
            3000,
            3500,
            4000,
            4500,
            5000,
        ]
        # budget_levels = [i for i in range(30, 150, 30)]
        # budget_levels = [i for i in range(500, 20500, 500)]
        return budget_levels
