# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import sys
import edmonds
import time
import os.path

import dynet_config

dynet_config.set(mem=2048, autobatch=False)
dynet_config.set_gpu()

import dynet as dy

from dynamic_network import DynamicNetwork
from embedding_network import EmbeddingNetwork


def store_model(model, path):
    print
    "Creating", path
    model.save_model(path)


def argmax(probs):
    max_index = 0
    for zz in range(len(probs)):
        if probs[zz] > probs[max_index]:
            max_index = zz
    return max_index


def eval_parsing(network, ds, config, log_output, train=False):
    duas = 0
    dlas = 0
    tuas = 0
    tlas = 0
    last_proc = 0
    # log_file = open(ds.dev_file+".log", "w")
    if train:
        sys.stdout.write(" train")
        for i in range(ds.num_train_sequences):
            cur_proc = (i * 100) / ds.num_train_sequences
            if cur_proc % 5 == 0 and cur_proc != last_proc:
                last_proc = cur_proc
                sys.stdout.write(" " + str(cur_proc))
                sys.stdout.flush()

            example = ds.get_next_train_sequence()
            dy.renew_cg()
            batch_x = ds.encode_input(example, True, config.use_morphology)

            predicted, predicted_labels = network.predict(batch_x)
            tree = edmonds.get_tree(predicted, len(example))
            for iSrc in range(len(example)):
                best_index = int(tree[iSrc])
                parts = example[iSrc].split("\t")
                best_label = "root"
                if best_index != -2 and best_index != 0:
                    label_probs = predicted_labels[iSrc][best_index].value()
                    label_index = argmax(label_probs)
                    best_label = ds.labels[label_index]

                label = best_label
                if best_index == 0:
                    label = "root"
                if iSrc != 0 and best_index == int(parts[6]):
                    tuas = tuas + 1
                    if label == parts[7]:
                        tlas += 1
    last_proc = 0
    sys.stdout.write(" dev")
    log_file = open(log_output, "w")
    for i in range(ds.num_dev_sequences):
        cur_proc = (i * 100) / ds.num_dev_sequences
        if cur_proc % 5 == 0 and cur_proc != last_proc:
            last_proc = cur_proc
            sys.stdout.write(" " + str(cur_proc))
            sys.stdout.flush()

        example = ds.get_next_dev_sequence()
        dy.renew_cg()
        batch_x = ds.encode_input(example, True, config.use_morphology)

        predicted, predicted_labels, aux_predicted, aux_predicted_labels, upos, xpos, attrs = network.predict(batch_x)
        tree = edmonds.get_tree(predicted, len(example))
        for iSrc in range(len(example)):
            best_index = int(tree[iSrc])
            parts = example[iSrc].split("\t")
            best_label = "root"
            if best_index != -2 and best_index != 0:
                label_probs = predicted_labels[iSrc][best_index].value()
                label_index = argmax(label_probs)
                best_label = ds.labels[label_index]

            label = best_label
            if best_index == 0:
                label = "root"
                # if has_root:
                #    print "errrorrrr"
                #    label="root***"
                # has_root=True
            if parts[6] != "_" and iSrc != 0 and best_index == int(parts[6]):
                duas = duas + 1
                if label.lower() == parts[7].lower():
                    dlas += 1
            import numpy as np
            str_upos = ds.upos_list[np.argmax(upos[iSrc].npvalue())]
            #print upos[iSrc].npvalue()
            str_xpos = ds.xpos_list[np.argmax(xpos[iSrc].npvalue())]
            str_attrs = ds.attrs_list[np.argmax(attrs[iSrc].npvalue())]
            if iSrc != 0:
                s = parts[0] + "\t" + parts[1] + "\t" + parts[
                    2] + "\t" + str_upos + "\t" + str_xpos + "\t" + str_attrs + "\t" + str(
                    best_index) + "\t" + label + "\t" + parts[8] + "\t" + parts[9]
                log_file.write(s + "\n")
        log_file.write("\n")
        log_file.flush()
    log_file.close()
    return float(tuas) / (ds.num_train_examples - ds.num_train_sequences), float(tlas) / (
            ds.num_train_examples - ds.num_train_sequences), float(duas) / (
                   ds.num_dev_examples - ds.num_dev_sequences), float(dlas) / (
                   ds.num_dev_examples - ds.num_dev_sequences)


def learn_sequence(network, ds, seq, config):
    batch_x = ds.encode_input(seq, False, config.use_morphology)
    prediction_y, prediction_y_labels, aux_prediction_y, aux_prediction_y_labels, upos, xpos, attrs = network.predict(
        batch_x, False)
    errors = 0
    # loss=0
    gold_heads = []
    gold_labels = []
    gold_upos = []
    gold_xpos = []
    gold_attrs = []
    for iSrc in range(len(seq)):
        parts = seq[iSrc].split("\t")
        to = int(parts[6])
        gold_heads.append(to)
        gold_labels.append(ds.label2int[parts[7]])
        gold_upos.append(parts[3])
        gold_xpos.append(parts[4])
        gold_attrs.append(parts[5])

    pred_heads = edmonds.get_tree(prediction_y, len(seq))
    for zz in range(1, len(seq)):
        if pred_heads[zz] != gold_heads[zz]:
            errors += 1
    # print pred_heads
    # print gold_heads
    loss, loss_source = network.learn(prediction_y, prediction_y_labels, aux_prediction_y, aux_prediction_y_labels,
                                      upos, xpos, attrs, pred_heads, gold_heads, gold_labels, gold_upos, gold_xpos,
                                      gold_attrs)

    return errors, loss, loss_source


def train(network, ds, num_itt_no_improve, output_base, config):
    best_uas = 0
    best_las = 0
    itt_no_improve = num_itt_no_improve
    epoch = 0
    batch_size = 5
    # store_model(network, output_base + ".last")
    # store_model(network, output_base + ".bestUAS")
    # store_model(network, output_base + ".bestLAS")
    # sys.stdout.write("Running initial evaluation")
    # tuas, tlas, duas, dlas = eval_parsing (network, ds, config, output_base + ".dev.log")
    # best_las = dlas
    # best_uas = duas
    # sys.stdout.write (" (train UAS=" + str(tuas) + " LAS=" + str(tlas) + " dev UAS=" + str(duas) + " LAS=" + str(dlas) + ")\n")

    while itt_no_improve > 0:
        itt_no_improve = itt_no_improve - 1
        sys.stdout.write("Epoch " + str(epoch) + "\n")
        sys.stdout.flush()
        epoch = epoch + 1

        # eval_parsing (network, ds)
        errors = 0
        words = 0
        loss = 0
        loss_sources = []
        epoch_start_time = time.time()
        start_time = time.time()
        for i in range(ds.num_train_sequences):
            # sys.stdout.write("\r"+str(i+1)+" sentences")
            seq = ds.get_next_train_sequence()
            # if len(seq) < 10:
            err, lss, loss_source = learn_sequence(network, ds, seq, config)
            loss_sources.append(loss_source)
            errors += err
            loss += lss
            words += len(seq) - 1

            if i % batch_size == 0:
                loss += network.update_batch(loss_sources)
                loss_sources = []
                dy.renew_cg()

            if i % 100 == 0:
                stop_time = time.time()
                elapsed_time = stop_time - start_time
                start_time = stop_time
                sys.stdout.write("Current stats: " + str(i + 1) + " sentences with " + str(
                    float(errors) / words) + " errors and loss=" + str(
                    loss / words) + " since last report EXEC: " + str(elapsed_time) + "\n")
                sys.stdout.flush()
                errors = 0
                words = 0
                loss = 0
        if len(loss_sources) != 0:
            network.update_batch(loss_sources)
            loss_sources = []
            dy.renew_cg()
        elapsed_time = time.time() - epoch_start_time
        sys.stdout.write("Epoch execution time is " + str(elapsed_time))
        sys.stdout.write("Evaluating on devset")
        sys.stdout.flush()
        tuas, tlas, duas, dlas = eval_parsing(network, ds, config, output_base + ".dev.log")
        sys.stdout.write(
            " (train UAS=" + str(tuas) + " LAS=" + str(tlas) + " dev UAS=" + str(duas) + " LAS=" + str(dlas) + ")\n")
        store_model(network, output_base + ".last")
        if duas > best_uas:
            best_uas = duas
            itt_no_improve = num_itt_no_improve
            store_model(network, output_base + ".bestUAS")

        if dlas > best_las:
            best_las = dlas
            itt_no_improve = num_itt_no_improve
            store_model(network, output_base + ".bestLAS")
    sys.stdout.write("\nDone with best UAS=" + str(best_uas) + " and best LAS=" + str(best_las))


def create_parsing_network(ds, configParam):
    network = DynamicNetwork(ds, configParam)

    return network


from dataset import Dataset
from config import Config
from dynamic_network import CharacterNetwork


def precompute_char_embeddings(ds, emb_size, cfg, parsing_network, output_base):
    sys.stdout.write("Precomputing character level embeddings\n")
    sys.stdout.flush()
    cn = CharacterNetwork(ds, emb_size)
    cn.init_embeddings(ds, cfg.embeddings_pretrain_epochs)
    cn.precache_embeddings(ds, parsing_network)
    cn.store_model(output_base)


def display_help():
    print("Neural tagger version 0.9 beta.")
    print("Usage:")
    print("\t--train <train file> <dev file> <model output base> <num itt no improve>")
    print("\t--test <model output base> <tokenized file>")


def do_train(train_file, dev, embeddings, output_base, num_itt_no_improve):
    cfg = Config(None)

    if not os.path.isfile(dev):
        dev = train_file
        print "Warning, dev file is missing so we are using the train file as dev... (overfit alert)"
    
    ds = Dataset(train_file, dev, embeddings, cfg.embeddings_size, cfg.use_morphology)
    network = create_parsing_network(ds, cfg)
    print "USE=", cfg.use_morphology
    if cfg.use_morphology == False:
        precompute_char_embeddings(ds, cfg.embeddings_size, cfg, network, output_base)

    ds.store_embeddings(output_base + ".embeddings")
    ds.store_lookups(output_base + ".encodings")

    train(network, ds, num_itt_no_improve, output_base, cfg)


def compute_char_embeddings(ds, emb_net, model_path, parsing_network):
    emb_net.load_model(model_path + "/parser")
    emb_net.precache_embeddings(ds, parsing_network)


def test(model_path, test_file, output_file=None):
    if output_file is None:
        output_file = test_file + ".out"
    cfg = Config(None)
    ds = Dataset(test_file, test_file, None, cfg.embeddings_size, cfg.use_morphology)
    ds.restore_lookups(model_path + "/parser.encodings")
    network = create_parsing_network(ds, cfg)
    if cfg.use_morphology == False:
        dummy_word_char_lookup = network.model.add_lookup_parameters(
            (len(ds.word2vec_character2int), ds.embeddings_size))
    network.load_model(model_path + "/parser.bestUAS")
    if cfg.use_morphology == False:
        emb_net = CharacterNetwork(ds, cfg.embeddings_size)
        compute_char_embeddings(ds, emb_net, model_path, network)

    # rebuild word embeddings
    ds.word2vec = ds.word2vecTmp
    ds.word2int = ds.word2intTmp
    # move into lookup parameters
    ds.lookup_word = network.model.add_lookup_parameters((len(ds.word2vec), ds.word_embeddings_size))
    for word in ds.word2vec:
        ds.lookup_word.init_row(ds.word2int[word], ds.word2vec[word])

    ign1, ign2, uas, las = eval_parsing(network, ds, cfg, output_file, False)
    print
    " UAS =", uas, "LAS =", las

if __name__=='__main__':
    if len(sys.argv) == 1:
        display_help()
    else:
        if sys.argv[1] == "--train" and len(sys.argv) == 5:
            os.mkdir(os.path.dirname(sys.argv[4]))
            do_train(sys.argv[2], sys.argv[3], None, sys.argv[4], 5)
        else:
            if (sys.argv[1] == "--test" and len(sys.argv) == 4):
                test(sys.argv[2], sys.argv[3])
            else:
                display_help()
