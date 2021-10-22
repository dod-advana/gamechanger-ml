import numpy as np
import pickle
import xgboost as xgb
import graphviz
import matplotlib
import math


class LTR:
    def __init__(
        self,
        data="xgboost.txt",
        params={"max_depth": 8, "eta": 1,
                "silent": 1, "objective": "reg:linear"},
        num_round=5,
    ):
        self.data = xgb.DMatrix(data)
        self.params = params
        self.num_round = num_round

    def write_model(self, model):
        with open("xgb-model.json", "w") as output:
            output.write("[" + ",".join(list(model)) + "]")
            output.close()

    def train(self, write=True):
        bst = xgb.train(self.params, self.data, self.num_round)
        model = bst.get_dump(fmap="featmap.txt", dump_format="json")
        if write:
            write_model(model)
        return bst, model
