# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

class Config:
    def __init__(self, filename=None):
        self.bdlstm_1_size = 100
        self.bdlstm_2_size = 100
        self.bdlstm_3_size = 100
        self.bdlstm_4_size = 100
        self.bdlstm_1_dropout = 0.33
        self.bdlstm_2_dropout = 0.33
        self.bdlstm_3_dropout = 0.33
        self.bdlstm_4_dropout = 0.33
        self.aux_softmax_weight = 0.2
        self.n_dimensional_projections_arc = 100
        self.n_dimensional_projections_label = 200
        self.embeddings_size = 100
        self.attention_dropout = 0.3
        self.use_morphology=False
        self.embeddings_pretrain_epochs=10

        
        if filename != None:
            with open(filename) as f:
                for line in f:
                    line = line.replace("\n", "").replace("\r", "")
                    if not line.startswith("#"):
                        parts = line.split(" ")
                        key = parts[0]
                        if key == "embeddings_size".upper():
                            self.embeddings_size = int(parts[1])
                        if key == "embeddings_pretain_epochs".upper():
                            self.embeddings_pretain_epochs = int(parts[1])
                        if key == "bdlstm_1_size".upper():
                            self.bdlstm_1_size = int(parts[1])
                        if key == "bdlstm_2_size".upper():
                            self.bdlstm_2_size = int(parts[1])
                        if key == "bdlstm_3_size".upper():
                            self.bdlstm_3_size = int(parts[1])
                        if key == "bdlstm_4_size".upper():
                            self.bdlstm_4_size = int(parts[1])
                        if key == "bdlstm_1_dropout".upper():
                            self.bdlstm_1_dropout = float(parts[1])
                        if key == "bdlstm_2_dropout".upper():
                            self.bdlstm_2_dropout = float(parts[1])
                        if key == "bdlstm_3_dropout".upper():
                            self.bdlstm_3_dropout = float(parts[1])
                        if key == "bdlstm_4_dropout".upper():
                            self.bdlstm_4_dropout = float(parts[1])
                        if key == "projection_size_arc".upper():
                            self.n_dimensional_projections_arc = int(parts[1])
                        if key == "projection_size_label".upper():
                            self.n_dimensional_projections_label = int(parts[1])
                        if key == "aux_softmax_weight".upper():
                            self.aux_softmax_weight = float(parts[1])
                        if key == "attention_dropout".upper():
                            self.attention_dropout = float(parts[1])
                        if key == "USE_MORPHOLOGY".upper():
                            self.use_morphology = (parts[1]=='True')
                            
    def store(self, path):
        f = open(path, "w")
        f.write("EMBEDDINGS_SIZE " + str(self.embeddings_size) + "\n")
        f.write("TANH_1_SIZE " + str(self.tanh_1_size) + "\n")
        f.write("TANH_2_SIZE " + str(self.tanh_2_size) + "\n")
        f.write("BDSLTM_1_SIZE " + str(self.bdlstm_1_size) + "\n")
        f.write("BDLSTM_2_SIZE " + str(self.bdlstm_2_size) + "\n")
        f.write("PROJECTION_SIZE " + str(self.projection_size) + "\n")
        f.close()