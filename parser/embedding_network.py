# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import dynet as dy
import numpy as np
import sys
class EmbeddingNetwork:
    def __init__(self, ds, itt, morph_embeddings_size):
        return
        sys.stdout.write("Initializing morphological embeddings networks\n")
        hidden_size = morph_embeddings_size
        output_size = ds.aux_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.upos_input_size = len(ds.upos2int)
        self.xpos_input_size = len(ds.xpos2int)
        self.uposModel = dy.Model()
        self.uposHiddenW = self.uposModel.add_parameters((hidden_size, len(ds.upos2int)))
        self.uposHiddenB = self.uposModel.add_parameters((hidden_size))
        self.uposOutputW = self.uposModel.add_parameters((output_size, hidden_size))
        self.uposOutputB = self.uposModel.add_parameters((output_size))
        
        self.xposModel = dy.Model()
        self.xposHiddenW = self.uposModel.add_parameters((hidden_size, len(ds.xpos2int)))
        self.xposHiddenB = self.uposModel.add_parameters((hidden_size))
        self.xposOutputW = self.uposModel.add_parameters((output_size, hidden_size))
        self.xposOutputB = self.uposModel.add_parameters((output_size))
        
        self.adamXpos = dy.AdamTrainer(self.xposModel)
        self.adamUpos = dy.AdamTrainer(self.uposModel)
        
        sys.stdout.write("Precomputing morphological embeddings for " + str(itt) + " itterations\n")
        
        for it in range(itt):
            total_loss_upos = 0
            total_loss_xpos = 0
            sys.stdout.write("Epoch " + str(it) + ":")
            sys.stdout.flush()
            last_proc = 0
            for iSeq in range (ds.num_train_sequences):
                example = ds.get_next_train_sequence()
                #batch_x, batch_y=ds.encode_input(example, True, True)
                cur_proc = iSeq * 100 / ds.num_train_sequences
                if cur_proc % 5 == 0 and cur_proc != last_proc:
                    last_proc = cur_proc
                    sys.stdout.write(" " + str(cur_proc))
                    sys.stdout.flush()
                    upos_loss, xpos_loss = self.update_embeddings(example, ds)
                    total_loss_xpos += xpos_loss
                    total_loss_upos += upos_loss
            sys.stdout.write(" 100 upos_loss=" + str(total_loss_upos) + " xpos_loss=" + str(total_loss_xpos) + "\n")
            
        sys.stdout.write("Creating embeddings lookup tables...")
        sys.stdout.flush()
        for upos in ds.upos2int:
            ds.upos2vec[upos] = self.embed(ds.upos2int[upos], len(ds.upos2int), self.uposHiddenW, self.uposHiddenB).value()
        for xpos in ds.xpos2int:
            ds.xpos2vec[xpos] = self.embed(ds.xpos2int[xpos], len(ds.xpos2int), self.xposHiddenW, self.xposHiddenB).value()
        ds.morph_embeddings_size = self.hidden_size
        ds.input_size = ds.word_embeddings_size + 2 * self.hidden_size
        sys.stdout.write("\n")
    
    def embed(self, input, input_size, hiddenW, hiddenB):
        x = dy.vecInput(input_size)
        xx = np.zeros(input_size)
        xx[input] = 1
        x.set(xx)
        hidden = dy.tanh(hiddenW.expr() * x + hiddenB.expr())
        return hidden
    
    def forward(self, input, input_size, hiddenW, hiddenB, outputW, outputB):
        dy.renew_cg()
        x = dy.vecInput(input_size)
        xx = np.zeros(input_size)
        xx[input] = 1
        x.set(xx)
        hidden = dy.tanh(hiddenW.expr() * x + hiddenB.expr())
        output = dy.logistic(outputW.expr() * hidden + outputB.expr())
        return output
        
        
    def backward(self, y_pred, y_target):
        y_t = dy.vecInput(self.output_size)
        y_t.set(y_target)
        loss = dy.squared_distance(y_pred, y_t)
        return loss
    
    def update_embeddings(self, example, ds):
        loss_upos = 0
        loss_xpos = 0
        batch_x, batch_y = ds.encode_input(example, True, True)
        for zz in range(len(example)):
            parts = example[zz].split("\t")
            upos = parts[3]
            xpos = parts[4]
            output = self.forward(ds.upos2int[upos], self.upos_input_size, self.uposHiddenW, self.uposHiddenB, self.uposOutputW, self.uposOutputB)
            l = self.backward(output, batch_y[0][zz])
            loss_upos += l.value()
            l.backward()
            
            output = self.forward(ds.xpos2int[xpos], self.xpos_input_size, self.xposHiddenW, self.xposHiddenB, self.xposOutputW, self.xposOutputB)
            l = self.backward(output, batch_y[0][zz])
            loss_xpos += l.value()
            l.backward()

            self.adamUpos.update()
        return loss_upos, loss_xpos
    
    