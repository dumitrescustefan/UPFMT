# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from config import Config
from dataset import Dataset
import dynet as dy
import dynet_config
import numpy as np
import random


class DynamicNetwork:
    def __init__(self, ds, configParam):
        self.lstm1_size = 2 * configParam.bdlstm_1_size
        self.lstm2_size = 2 * configParam.bdlstm_2_size
        self.lstm3_size = 2 * configParam.bdlstm_3_size
        self.lstm4_size = 2 * configParam.bdlstm_4_size
        self.n_proj_arc = configParam.n_dimensional_projections_arc
        self.n_proj_label = configParam.n_dimensional_projections_label
        self.lstm1_dropout = configParam.bdlstm_1_dropout
        self.lstm2_dropout = configParam.bdlstm_2_dropout
        self.lstm3_dropout = configParam.bdlstm_3_dropout
        self.lstm4_dropout = configParam.bdlstm_4_dropout
        self.aux_softmax_weight = configParam.aux_softmax_weight
        self.attention_dropout = configParam.attention_dropout
        self.dataset = ds

        self.model = dy.Model()

        self.trainer = dy.AdamTrainer(self.model, alpha=0.001)
        # self.trainer=dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)

        self.lstm_1_forward = dy.LSTMBuilder(1, ds.input_size, configParam.bdlstm_1_size, self.model)
        self.lstm_1_backward = dy.LSTMBuilder(1, ds.input_size, configParam.bdlstm_1_size, self.model)

        self.lstm_2_forward = dy.LSTMBuilder(1, self.lstm1_size, configParam.bdlstm_2_size, self.model)
        self.lstm_2_backward = dy.LSTMBuilder(1, self.lstm1_size, configParam.bdlstm_2_size, self.model)

        self.upos_softmax_w = self.model.add_parameters((len(ds.upos2int), configParam.bdlstm_2_size * 2))
        self.upos_softmax_b = self.model.add_parameters((len(ds.upos2int)))
        self.xpos_softmax_w = self.model.add_parameters((len(ds.xpos2int), configParam.bdlstm_2_size * 2))
        self.xpos_softmax_b = self.model.add_parameters((len(ds.xpos2int)))
        self.attrs_softmax_w = self.model.add_parameters((len(ds.attrs2int), configParam.bdlstm_2_size * 2))
        self.attrs_softmax_b = self.model.add_parameters((len(ds.attrs2int)))

        self.lstm_3_forward = dy.LSTMBuilder(1, self.lstm2_size, configParam.bdlstm_3_size, self.model)
        self.lstm_3_backward = dy.LSTMBuilder(1, self.lstm2_size, configParam.bdlstm_3_size, self.model)

        self.lstm_4_forward = dy.LSTMBuilder(1, self.lstm3_size, configParam.bdlstm_4_size, self.model)
        self.lstm_4_backward = dy.LSTMBuilder(1, self.lstm3_size, configParam.bdlstm_4_size, self.model)

        # main output
        self.dep_to_w = self.model.add_parameters(
            (configParam.n_dimensional_projections_arc, configParam.bdlstm_4_size * 2))
        self.dep_from_w = self.model.add_parameters(
            (configParam.n_dimensional_projections_arc, configParam.bdlstm_4_size * 2))
        self.dep_to_w_lab = self.model.add_parameters(
            (configParam.n_dimensional_projections_label, configParam.bdlstm_4_size * 2))
        self.dep_from_w_lab = self.model.add_parameters(
            (configParam.n_dimensional_projections_label, configParam.bdlstm_4_size * 2))
        self.dep_link_b = self.model.add_parameters((configParam.n_dimensional_projections_arc))
        self.dep_label_b = self.model.add_parameters((configParam.n_dimensional_projections_label))

        self.link_attention_W1 = self.model.add_parameters(
            (configParam.n_dimensional_projections_arc, configParam.n_dimensional_projections_arc))
        self.link_attention_W2 = self.model.add_parameters((1, configParam.n_dimensional_projections_arc))

        self.outputW = self.model.add_parameters((1, configParam.n_dimensional_projections_arc))
        self.outputB = self.model.add_parameters((1, configParam.n_dimensional_projections_arc))

        self.output_w1_labels = [self.model.add_parameters(
            (configParam.n_dimensional_projections_label, configParam.n_dimensional_projections_label)) for _ in
                                 range(len(ds.label2int) - 1)]
        self.output_w2_labels = self.model.add_parameters(
            (len(ds.label2int) - 1, configParam.n_dimensional_projections_label * 2))
        self.output_b_labels = self.model.add_parameters((len(ds.label2int) - 1))

        # aux softmax
        self.aux_dep_to_w = self.model.add_parameters(
            (configParam.n_dimensional_projections_arc, configParam.bdlstm_2_size * 2))
        self.aux_dep_from_w = self.model.add_parameters(
            (configParam.n_dimensional_projections_arc, configParam.bdlstm_2_size * 2))
        self.aux_dep_to_w_lab = self.model.add_parameters(
            (configParam.n_dimensional_projections_label, configParam.bdlstm_2_size * 2))
        self.aux_dep_from_w_lab = self.model.add_parameters(
            (configParam.n_dimensional_projections_label, configParam.bdlstm_2_size * 2))
        self.aux_dep_link_b = self.model.add_parameters((configParam.n_dimensional_projections_arc))
        self.aux_dep_label_b = self.model.add_parameters((configParam.n_dimensional_projections_label))

        self.aux_link_attention_W1 = self.model.add_parameters(
            (configParam.n_dimensional_projections_arc, configParam.n_dimensional_projections_arc))
        self.aux_link_attention_W2 = self.model.add_parameters((1, configParam.n_dimensional_projections_arc))

        self.aux_outputW = self.model.add_parameters((1, configParam.n_dimensional_projections_arc))
        self.aux_outputB = self.model.add_parameters((1, configParam.n_dimensional_projections_arc))

        self.aux_output_w1_labels = [self.model.add_parameters(
            (configParam.n_dimensional_projections_label, configParam.n_dimensional_projections_label)) for _ in
                                     range(len(ds.label2int) - 1)]
        self.aux_output_w2_labels = self.model.add_parameters(
            (len(ds.label2int) - 1, configParam.n_dimensional_projections_label * 2))
        self.aux_output_b_labels = self.model.add_parameters((len(ds.label2int) - 1))

        self.num_labels = len(ds.label2int) - 1

        # ds.lookup_word = self.model.add_lookup_parameters((len(ds.word2vec), ds.word_embeddings_size))
        # ds.lookup_upos = self.model.add_lookup_parameters((len(ds.upos2int), configParam.embeddings_size))
        # ds.lookup_xpos = self.model.add_lookup_parameters((len(ds.xpos2int), configParam.embeddings_size))
        # ds.lookup_attrs = self.model.add_lookup_parameters((len(ds.attrs2int), configParam.embeddings_size))
        # print "ds.upos2int", len(ds.upos2int)
        # print "ds.xpos2int", len(ds.xpos2int)
        # for word in ds.word2vec:
        #    ds.lookup_word.init_row(ds.word2int[word], ds.word2vec[word])

    def relu(self, x):
        EPS = 0.01
        ex = -EPS * x
        return dy.bmax(x, ex)

    def predict(self, batch_x, runtime=True):

        input_list = []
        for zz in range(len(batch_x)):
            x = batch_x[zz]
            input_list.append(x)
        lstm_forward = self.lstm_1_forward.initial_state()
        lstm_backward = self.lstm_1_backward.initial_state()
        # print lstm_forward
        if runtime:
            self.lstm_1_forward.set_dropouts(0, 0)
            self.lstm_1_backward.set_dropouts(0, 0)
            self.lstm_2_forward.set_dropouts(0, 0)
            self.lstm_2_backward.set_dropouts(0, 0)
            self.lstm_3_forward.set_dropouts(0, 0)
            self.lstm_3_backward.set_dropouts(0, 0)
            self.lstm_4_forward.set_dropouts(0, 0)
            self.lstm_4_backward.set_dropouts(0, 0)
        else:
            self.lstm_1_forward.set_dropouts(0, self.lstm1_dropout)
            self.lstm_1_backward.set_dropouts(0, self.lstm1_dropout)
            self.lstm_2_forward.set_dropouts(0, self.lstm2_dropout)
            self.lstm_2_backward.set_dropouts(0, self.lstm2_dropout)
            self.lstm_3_forward.set_dropouts(0, self.lstm3_dropout)
            self.lstm_3_backward.set_dropouts(0, self.lstm3_dropout)
            self.lstm_4_forward.set_dropouts(0, self.lstm4_dropout)
            self.lstm_4_backward.set_dropouts(0, self.lstm4_dropout)

        lstm1_forward_output = lstm_forward.transduce(input_list)
        lstm1_backward_output = list(reversed(lstm_backward.transduce(reversed(input_list))))

        lstm_1_list = []
        zero = dy.vecInput(self.lstm1_size)
        zero.set([0] * self.lstm1_size)
        for zz in range(len(batch_x)):
            x = dy.concatenate([lstm1_forward_output[zz], lstm1_backward_output[zz]])
            lstm_1_list.append(x)

        lstm_forward = self.lstm_2_forward.initial_state()
        lstm_backward = self.lstm_2_backward.initial_state()

        lstm2_forward_output = lstm_forward.transduce(lstm_1_list)
        lstm2_backward_output = list(reversed(lstm_backward.transduce(reversed(lstm_1_list))))

        lstm_2_list = []
        for zz in range(len(batch_x)):
            lstm_2_list.append(dy.concatenate([lstm2_forward_output[zz], lstm2_backward_output[zz]]))

        lstm_forward = self.lstm_3_forward.initial_state()
        lstm_backward = self.lstm_3_backward.initial_state()

        lstm3_forward_output = lstm_forward.transduce(lstm_2_list)
        lstm3_backward_output = list(reversed(lstm_backward.transduce(reversed(lstm_2_list))))
        lstm_3_list = []
        for zz in range(len(batch_x)):
            lstm_3_list.append(dy.concatenate([lstm3_forward_output[zz], lstm3_backward_output[zz]]))

        lstm_forward = self.lstm_4_forward.initial_state()
        lstm_backward = self.lstm_4_backward.initial_state()

        lstm4_forward_output = lstm_forward.transduce(lstm_3_list)
        lstm4_backward_output = list(reversed(lstm_backward.transduce(reversed(lstm_3_list))))

        dep_to_activ = []
        dep_from_activ = []
        dep_to_activ_lab = []
        dep_from_activ_lab = []

        aux_dep_to_activ = []
        aux_dep_from_activ = []
        aux_dep_to_activ_lab = []
        aux_dep_from_activ_lab = []

        zero = dy.vecInput(self.lstm2_size)
        zero.set([0] * self.lstm2_size)

        for zz in range(len(batch_x)):

            x = dy.concatenate([lstm4_forward_output[zz], lstm4_backward_output[zz]])
            dropout_rate = 0
            if not runtime:
                dropout_rate = self.attention_dropout
            dep_to_activ.append(dy.dropout(dy.rectify(self.dep_to_w.expr() * x), dropout_rate))
            dep_from_activ.append(dy.dropout(dy.rectify(self.dep_from_w.expr() * x), dropout_rate))
            dep_to_activ_lab.append(dy.dropout(dy.rectify(self.dep_to_w_lab.expr() * x), dropout_rate))
            dep_from_activ_lab.append(dy.dropout(dy.rectify(self.dep_from_w_lab.expr() * x), dropout_rate))
            ##aux softmax
            x = dy.concatenate([lstm2_forward_output[zz], lstm2_backward_output[zz]])
            aux_dep_to_activ.append(dy.dropout(dy.rectify(self.aux_dep_to_w.expr() * x), dropout_rate))
            aux_dep_from_activ.append(dy.dropout(dy.rectify(self.aux_dep_from_w.expr() * x), dropout_rate))
            aux_dep_to_activ_lab.append(dy.dropout(dy.rectify(self.aux_dep_to_w_lab.expr() * x), dropout_rate))
            aux_dep_from_activ_lab.append(dy.dropout(dy.rectify(self.aux_dep_from_w_lab.expr() * x), dropout_rate))

        output = []
        output_labels = []
        aux_output = []
        aux_output_labels = []

        # expression precaching
        pre_output_w_label = []
        aux_pre_output_w_label = []
        pre_output_w_arc = []
        aux_pre_output_w_arc = []
        pre_output_b_arc = []
        aux_pre_output_b_arc = []
        # pre_output_w_
        for i in range(len(batch_x)):
            w_tmp = [w1.expr() * dep_from_activ_lab[i] for w1 in
                     self.output_w1_labels]  # [dy.reshape(dy.reshape(w1.expr() * dep_from_activ_lab[i], (1, self.n_proj_label)), ()) for w1 in self.output_w1_labels]
            pre_output_w_label.append(dy.transpose(dy.concatenate_cols(w_tmp)))
            weight = dy.reshape(self.link_attention_W1.expr() * dep_from_activ[i], (1, self.n_proj_arc))
            pre_output_w_arc.append(weight)
            pre_output_b_arc.append(self.link_attention_W2.expr() * dep_from_activ[i])

            w_tmp = [w1.expr() * aux_dep_from_activ_lab[i] for w1 in
                     self.aux_output_w1_labels]  # [dy.reshape(dy.reshape(w1.expr() * dep_from_activ_lab[i], (1, self.n_proj_label)), ()) for w1 in self.output_w1_labels]
            aux_pre_output_w_label.append(dy.transpose(dy.concatenate_cols(w_tmp)))
            weight = dy.reshape(self.aux_link_attention_W1.expr() * aux_dep_from_activ[i], (1, self.n_proj_arc))
            aux_pre_output_w_arc.append(weight)
            aux_pre_output_b_arc.append(self.link_attention_W2.expr() * aux_dep_from_activ[i])

        upos = []
        xpos = []
        attrs = []
        for iSrc in range(len(batch_x)):
            upos.append(dy.softmax(self.upos_softmax_w.expr() * lstm_2_list[iSrc] + self.upos_softmax_b.expr()))
            xpos.append(dy.softmax(self.xpos_softmax_w.expr() * lstm_2_list[iSrc] + self.xpos_softmax_b.expr()))
            attrs.append(dy.softmax(self.attrs_softmax_w.expr() * lstm_2_list[iSrc] + self.attrs_softmax_b.expr()))

        for iSrc in range(len(batch_x)):
            output_row = []
            output_row_labels = []
            aux_output_row = []
            aux_output_row_labels = []
            for iDst in range(len(batch_x)):
                # if iSrc != iDst:
                dropout_rate = 0
                if not runtime:
                    dropout_rate = self.attention_dropout
                # arc
                w1 = dy.parameter(self.link_attention_W1)
                w2 = dy.parameter(self.link_attention_W2)

                weight = pre_output_w_arc[iDst]
                bias = pre_output_b_arc[iDst]
                output_row.append(weight * dep_to_activ[iSrc] + bias)
                # label
                term1 = pre_output_w_label[iDst] * dep_to_activ_lab[iSrc]
                term2 = self.output_w2_labels.expr() * dy.concatenate(
                    [dep_from_activ_lab[iDst], dep_to_activ_lab[iSrc]])
                lab_prob = dy.softmax(self.output_b_labels.expr() + term1 + term2)
                output_row_labels.append(lab_prob)

                # aux softmax arc
                w1 = dy.parameter(self.aux_link_attention_W1)
                w2 = dy.parameter(self.aux_link_attention_W2)

                weight = aux_pre_output_w_arc[iDst]
                bias = aux_pre_output_b_arc[iDst]
                aux_output_row.append(weight * aux_dep_to_activ[iSrc] + bias)

                term1 = aux_pre_output_w_label[iDst] * aux_dep_to_activ_lab[iSrc]
                term2 = self.aux_output_w2_labels.expr() * dy.concatenate(
                    [aux_dep_from_activ_lab[iDst], aux_dep_to_activ_lab[iSrc]])
                lab_prob = dy.softmax(self.output_b_labels.expr() + term1 + term2)
                aux_output_row_labels.append(lab_prob)
                # else:
                #    output_row.append(dy.scalarInput(0))
                #    output_row_labels.append(None)
                #    aux_output_row.append(dy.scalarInput(0))
                #    aux_output_row_labels.append(None)
            # link_att_w=dy.softmax(dy.concatenate)
            output.append(output_row)
            output_labels.append(output_row_labels)
            aux_output.append(aux_output_row)
            aux_output_labels.append(aux_output_row_labels)

        return output, output_labels, aux_output, aux_output_labels, upos, xpos, attrs

    def argmax(self, vec):
        m = 0
        for zz in range(len(vec)):
            if vec[zz] > vec[m]:
                m = zz
        return m

    def learn(self, prediction, prediction_labels, aux_prediction, aux_prediction_labels, upos, xpos, attrs, pred_heads,
              gold_heads, gold_labels, gold_upos, gold_xpos, gold_attrs):
        losses = []

        for iSrc in range(1, len(gold_heads)):
            gh = gold_heads[iSrc]
            output_probs = dy.softmax(dy.concatenate(prediction[iSrc]))  # dy.softmax(dy.concatenate(prediction[iSrc]))
            loss_arc = -dy.log(dy.pick(output_probs, gold_heads[iSrc]))
            losses.append(loss_arc * (1.0))  # -self.aux_softmax_weight))
            output_probs_aux = dy.softmax(
                dy.concatenate(aux_prediction[iSrc]))  # dy.softmax(dy.concatenate(aux_prediction[iSrc]))
            # loss_arc_aux = -dy.log(dy.pick(output_probs_aux, gold_heads[iSrc]))
            # losses.append(loss_arc_aux * self.aux_softmax_weight)
            if gold_upos[iSrc] in self.dataset.upos2int:
                losses.append(
                    -dy.log(dy.pick(upos[iSrc], self.dataset.upos2int[gold_upos[iSrc]])) * self.aux_softmax_weight)
            if gold_xpos[iSrc] in self.dataset.xpos2int:
                losses.append(
                    -dy.log(dy.pick(xpos[iSrc], self.dataset.xpos2int[gold_xpos[iSrc]])) * self.aux_softmax_weight)
            if gold_attrs[iSrc] in self.dataset.attrs2int:
                losses.append(
                    -dy.log(dy.pick(attrs[iSrc], self.dataset.attrs2int[gold_attrs[iSrc]])) * self.aux_softmax_weight)

            if gh != iSrc:
                loss_labels = -dy.log(dy.pick(prediction_labels[iSrc][gh], gold_labels[
                    iSrc]))  # -log(pick(prediction_labels[iSrc][iDst], gold_labels[iSrc]))
                # loss_labels = dy.pickneglogsoftmax(prediction_labels[iSrc][gh], gold_labels[iSrc])
                losses.append(loss_labels * (1.0))  # -self.aux_softmax_weight))
                # loss_labels_aux = -dy.log(dy.pick(aux_prediction_labels[iSrc][gh], gold_labels[iSrc]))##-log(pick(prediction_labels[iSrc][iDst], gold_labels[iSrc]))
                # loss_labels_aux = dy.pickneglogsoftmax(aux_prediction_labels[iSrc][gh], gold_labels[iSrc])##-log(pick(prediction_labels[iSrc][iDst], gold_labels[iSrc]))
                # losses.append(loss_labels_aux * self.aux_softmax_weight)

        l = 0
        if len(losses) == 0:
            print
            pred_heads
            print
            gold_heads
            print
            "-----------"
        else:
            loss = dy.esum(losses)
            #l = loss.npvalue()
            # loss.backward()
            # self.trainer.update()

        return 0, loss

    def update_batch(self, batch_losses):
        loss = dy.esum(batch_losses)
        val=loss.value()
        loss.backward()
        self.trainer.update()
        return val

    def learn_disjoint(self, prediction, prediction_labels, aux_prediction, aux_prediction_labels, pred_heads,
                       gold_heads, gold_labels):
        if pred_heads is None:
            # print "errrorr"
            return 1
        losses = []

        hinge = 0.1
        scalar = dy.scalarInput(hinge)
        one = dy.scalarInput(1.0)
        for iSrc in range(1, len(gold_heads)):
            gh = gold_heads[iSrc]
            ph = int(pred_heads[iSrc])
            for iDst in range(1, len(gold_heads)):
                if iSrc != iDst:
                    y_pred = prediction[iSrc][iDst]
                    y_pred_aux = aux_prediction[iSrc][iDst]
                    if iDst == gh:
                        target = 1
                    else:
                        target = 0
                    y_gold = dy.scalarInput(target)
                    loss = dy.squared_distance(y_pred,
                                               y_gold)  # -y_gold*dy.log(y_pred)-(one-y_gold)*dy.log(one-y_pred)##-y_gold*log(y_pred)-(one-y_gold)*log(one-y_pred)
                    losses.append(loss)
                    # loss = dy.squared_distance(y_pred_aux, y_gold) * 0.2#-y_gold*dy.log(y_pred)-(one-y_gold)*dy.log(one-y_pred)##-y_gold*log(y_pred)-(one-y_gold)*log(one-y_pred)
                    # losses.append(loss)
                    # if iDst == pred_heads[iSrc] and iDst != gh:
                    #    loss = prediction[iSrc][iDst]-prediction[iSrc][gh] + scalar
                    #    losses.append(loss)

                    # labels
                    if gh == iDst:
                        lbl_target = dy.vecInput(self.num_labels)
                        tgt = [0] * self.num_labels
                        tgt[gold_labels[iSrc]] = 1
                        lbl_target.set(tgt)
                        loss_labels = dy.squared_distance(prediction_labels[iSrc][iDst],
                                                          lbl_target)  # -dy.log(dy.pick(prediction_labels[iSrc][iDst], gold_labels[iSrc]))##-log(pick(prediction_labels[iSrc][iDst], gold_labels[iSrc]))
                        losses.append(loss_labels)
                        # loss_labels = dy.squared_distance(aux_prediction_labels[iSrc][iDst], lbl_target) * 0.2#-dy.log(dy.pick(prediction_labels[iSrc][iDst], gold_labels[iSrc]))##-log(pick(prediction_labels[iSrc][iDst], gold_labels[iSrc]))
                        # losses.append(loss_labels)

        l = 0
        if len(losses) == 0:
            print
            pred_heads
            print
            gold_heads
            print
            "-----------"
            cnt = 0
        else:
            loss = dy.esum(losses)
            l = loss.npvalue()
            loss.backward()
            self.trainer.update()

        return l

    def learn_hinge(self, prediction, prediction_labels, aux_prediction, aux_prediction_labels, pred_heads, gold_heads,
                    gold_labels):
        if pred_heads is None:
            # print "errrorr"
            return 1
        losses = []

        hinge = 0.1
        scalar = dy.scalarInput(hinge)
        one = dy.scalarInput(1.0)
        zero = dy.scalarInput(0)
        for iSrc in range(1, len(gold_heads)):
            gh = gold_heads[iSrc]
            ph = int(pred_heads[iSrc])
            # arc loss
            loss = -(prediction[iSrc][gh] - (prediction[iSrc][ph] + scalar))
            losses.append(loss)
            # loss = dy.squared_distance(aux_prediction[iSrc][gh], one) * 0.2
            # losses.append(loss)
            # loss = dy.squared_distance(prediction[iSrc][gh], one)
            # losses.append(loss)
            # if gh != ph and ph != iSrc:
            #    loss = dy.squared_distance(prediction[iSrc][ph], zero)
            #    losses.append(loss)
            # label error
            lbl_target = dy.vecInput(self.num_labels)
            tgt = [0] * self.num_labels
            tgt[gold_labels[iSrc]] = 1
            lbl_target.set(tgt)
            loss_labels = dy.squared_distance(prediction_labels[iSrc][gh], lbl_target)
            losses.append(loss_labels)
            loss_labels = dy.squared_distance(aux_prediction_labels[iSrc][gh], lbl_target) * 0.2
            losses.append(loss_labels)

        l = 0
        if len(losses) == 0:
            print
            pred_heads
            print
            gold_heads
            print
            "-----------"
            cnt = 0
        else:
            loss = dy.esum(losses)
            l = loss.npvalue()
            loss.backward()
            self.trainer.update()
        return l

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.populate(path)


import re
import sys


class CharacterNetwork:
    def __init__(self, ds, embeddings_size):
        self.embeddings_size = embeddings_size
        self.character2int = {}
        self.ds = ds

        for i in range(ds.num_train_sequences):
            example = ds.get_next_train_sequence()
            for line in example:
                parts = line.split("\t")
                word = parts[1]
                chars = self.get_characters(word)
                for char in chars:
                    if char not in self.character2int:
                        self.character2int[char] = len(self.character2int)

        sys.stdout.write("Found " + str(len(self.character2int)) + " unique characters\n")
        print
        self.character2int

        self.init_model()

    def init_model(self):
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        self.char_lookup = self.model.add_lookup_parameters((len(self.character2int), 32))
        self.lstm = dy.VanillaLSTMBuilder(1, 32, self.embeddings_size, self.model)

        self.uposW = self.model.add_parameters((len(self.ds.upos2int), self.embeddings_size))
        self.uposB = self.model.add_parameters((len(self.ds.upos2int)))
        self.xposW = self.model.add_parameters((len(self.ds.xpos2int), self.embeddings_size))
        self.xposB = self.model.add_parameters((len(self.ds.xpos2int)))

    def predict(self, word):
        chars = self.get_characters(word)
        lstm_fw = self.lstm.initial_state()

        for zz in range(len(chars)):
            char = chars[zz]
            if char in self.character2int:
                emb_char = self.char_lookup[self.character2int[char]]
            else:
                emb_char = dy.vecInput(32)
                emb_char.set(np.zeros(32))
            lstm_fw = lstm_fw.add_input(emb_char)
        s_upos = dy.softmax(self.uposW.expr() * lstm_fw.output() + self.uposB.expr())
        s_xpos = dy.softmax(self.xposW.expr() * lstm_fw.output() + self.xposB.expr())
        return s_upos, s_xpos, lstm_fw

    def learn(self, s_upos, s_xpos, upos, xpos, ds):
        loss_upos = -dy.log(dy.pick(s_upos, ds.upos2int[upos]))
        loss_xpos = -dy.log(dy.pick(s_xpos, ds.xpos2int[xpos]))
        return loss_upos + loss_xpos

    def argmax(self, x):
        vals = x.value()
        max = 0
        for zz in range(1, len(vals)):
            if vals[zz] > vals[max]:
                max = zz
        return max

    def precache_embeddings(self, ds, parsing_network):
        ds.word2vec_character = {}
        ds.word2vec_character2int = {}
        sys.stdout.write("Precaching embeddings...")
        sys.stdout.flush()
        for iSeq in range(ds.num_train_sequences):
            example = ds.get_next_train_sequence()
            for line in example:
                parts = line.split("\t")
                word = parts[1]
                if word not in ds.word2vec_character:
                    dy.renew_cg()
                    u, x, lstm = self.predict(word)
                    embeddings = lstm.output().value()
                    ds.word2vec_character[word] = embeddings
                    ds.word2vec_character2int[word] = len(ds.word2vec_character2int)
        for iSeq in range(ds.num_dev_sequences):
            example = ds.get_next_dev_sequence()
            for line in example:
                parts = line.split("\t")
                word = parts[1]
                if word not in ds.word2vec_character:
                    dy.renew_cg()
                    u, x, lstm = self.predict(word)
                    embeddings = lstm.output().value()
                    ds.word2vec_character[word] = embeddings
                    ds.word2vec_character2int[word] = len(ds.word2vec_character2int)
        # move data to lookup_param
        ds.word_char_lookup = parsing_network.model.add_lookup_parameters(
            (len(ds.word2vec_character2int), ds.embeddings_size))
        for word in ds.word2vec_character2int:
            ds.word_char_lookup.init_row(ds.word2vec_character2int[word], ds.word2vec_character[word])
        sys.stdout.write("done\n")

    def init_embeddings(self, ds, num_itt):
        print
        "Pretraining embeddings for " + str(num_itt) + " epochs"
        for zz in range(num_itt):
            sys.stdout.write("Epoch " + str(zz) + "...")
            sys.stdout.flush()
            last_proc = 0
            total_loss = 0
            for iSeq in range(ds.num_train_sequences):
                proc = iSeq * 100 / ds.num_train_sequences
                if proc != last_proc and proc % 5 == 0:
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()
                    last_proc = proc
                dy.renew_cg()
                example = ds.get_next_train_sequence()
                losses = []
                for line in example:
                    parts = line.split("\t")                    
                    word = parts[1]
                    upos = parts[3]
                    xpos = parts[4]
                    if upos in ds.upos2int and xpos in ds.xpos2int:
                        s_upos, s_xpos, lstm = self.predict(word)
                        loss = self.learn(s_upos, s_xpos, upos, xpos, ds)
                        losses.append(loss)
                ls = dy.esum(losses)
                total_loss += ls.value()
                ls.backward()
                self.trainer.update()

            upos_err = 0
            xpos_err = 0
            sys.stdout.write(" evaluating ")
            for iSeq in range(ds.num_dev_sequences):
                example = ds.get_next_dev_sequence()
                for iLine in range(1, len(example)):
                    dy.renew_cg()
                    parts = example[iLine].split("\t")
                    word = parts[1]
                    upos = parts[3]
                    xpos = parts[4]
                    upos_index = -1
                    xpos_index = -1
                    if upos in ds.upos2int:
                        upos_index = ds.upos2int[upos]
                    if xpos in ds.xpos2int:
                        xpos_index = ds.xpos2int[xpos]
                    s_upos, s_xpos, lstm = self.predict(word)
                    if self.argmax(s_upos) != upos_index:
                        upos_err += 1
                    if self.argmax(s_xpos) != xpos_index:
                        xpos_err += 1

            upos_err = float(upos_err) / (ds.num_dev_examples - ds.num_dev_sequences)
            xpos_err = float(xpos_err) / (ds.num_dev_examples - ds.num_dev_sequences)
            sys.stdout.write(
                "upos error=" + str(1.0 - upos_err) + " xpos error=" + str(1.0 - xpos_err) + " train loss=" + str(
                    total_loss) + "\n")

    def store_model(self, output_base):
        sys.stdout.write("Storing character-level embeddings network...")
        sys.stdout.flush()
        f = open(output_base + "-character.encodings", "w")
        f.write("NUM_CHARACTERS " + str(len(self.character2int)) + "\n")
        for char in self.character2int:
            f.write(char + " " + str(self.character2int[char]) + "\n")
        f.close()
        self.model.save(output_base + "-character.network")
        sys.stdout.write("done\n")

    def load_model(self, model_base):
        sys.stdout.write("Loading character-level embeddings network...")
        sys.stdout.flush()
        f = open(model_base + "-character.encodings", "r")
        line = f.readline().replace("\n", "")
        parts = line.split(" ")
        num_chars = int(parts[1])
        self.character2int = {}
        for zz in range(num_chars):
            parts = f.readline().replace("\n", "").split(" ")
            char = parts[0]
            index = int(parts[1])
            self.character2int[char] = index
        f.close()
        self.init_model()
        self.model.populate(model_base + "-character.network")
        sys.stdout.write("done\n")

    def get_characters(self, word):
        word = word.replace(" ", "_").lower()
        word = re.sub('\d', '0', word)
        chars = []
        uniword = unicode(word, 'utf-8')
        for i in range(len(uniword)):
            char = uniword[i].encode("utf-8")
            chars.append(char)

        return chars
