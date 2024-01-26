from NNPyTorchModel.ModelCdr import *

class ModelParser:
    model_op_class = None
    used = False
    def __init__(self, file_name) -> None:
        self.model_op_class = ModelOp(1,'',not_imagenet_class=True)
        self.model_op_class.load_model(file_name)

    def get_id_by_offset(self,i):
        if self.used:
            return 0
        else:
            self.used = True
            return 1 
    def remove_obj_by_id(self, i, rules):
        disarm_action = rules["weights"][1][0]
        if disarm_action == 'qint8':
            print("quantsize the model")
            self.model_op_class.disarm("quanti8",1)
        else:
            print("disarm using n-random", int(disarm_action))
            self.model_op_class.disarm("n_random",int(disarm_action))

    def dump_file(self,path, new_name = " "):
        new_file_name = path + "/" + new_name
        with open(new_file_name,"wb") as f:
            pickle.dump(self.model_op_class.model,f)