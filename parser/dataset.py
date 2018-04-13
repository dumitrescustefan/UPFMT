# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import dynet as dy
import numpy as np
import os
import random
import sys


class Dataset:
    # dynet embeddings
    lookup_word = None
    lookup_xpos = None
    lookup_upos = None

    word2vec = {}
    xpos2vec = {}
    upos2vec = {}
    word_embeddings_size = 0
    morph_embeddings_size = 0
    xpos2int = {}
    attrs2int = {}  # deocamdata nu folosim asa ceva
    upos2int = {}

    word2int = {}
    label2int = {}
    labels = None
    num_train_examples = 0
    num_train_sequences = 0
    num_dev_examples = 0
    num_dev_sequences = 0
    num_embeddings = 0
    input_size = 0
    output_size = 0
    aux_size = 0
    train_file = None
    dev_file = None
    file_train = None
    file_dev = None
    drop_prob = 0.33

    def __init__(self, train_file, dev_file, word_embeddings_file, embeddings_size, use_morph):
        self.embeddings_size = embeddings_size
        self.train_file = train_file
        self.dev_file = dev_file
        print("Initializing dataset reader")
        self.upos2int["root"] = 0
        self.xpos2int["root"] = 0
        self.attrs2int["root"] = 0
        self.label2int["None"] = 0
        self.upos_list=["root"]
        self.xpos_list=["root"]
        self.attrs_list=["root"]
        self.use_morph = use_morph
        trainableWE = {}
        trainableWE["</s>"] = 1
        with open(train_file) as f:
            for line in f:
                line = line.replace("\n", "").replace("\r", "")
                if not line.startswith("#"):
                    self.num_train_examples = self.num_train_examples + 1
                    if self.num_train_examples % 50000 == 0:
                        print("Read", self.num_train_examples, "train examples")
                    parts = line.split("\t")                  
                    if "-" not in parts[0]:
                        if (len(parts) != 1):
                            if "_" not in parts[6]:
                                word = parts[1].lower()
                                upos = parts[3]
                                xpos = parts[4]
                                attrs = parts[5]
                                label = parts[7]
                                if word not in trainableWE:
                                    trainableWE[word] = 1
                                if upos not in self.upos2int:
                                    self.upos2int[upos] = len(self.upos2int)
                                    self.upos_list.append(upos)
                                if attrs not in self.attrs2int:
                                    self.attrs2int[attrs] = len(self.attrs2int)
                                    self.attrs_list.append(attrs)
                                if xpos not in self.xpos2int:
                                    self.xpos_list.append(xpos)
                                    self.xpos2int[xpos] = len(self.xpos2int)
                                if label not in self.label2int:
                                    self.label2int[label] = len(self.label2int)                                
                        else:
                            self.num_train_sequences = self.num_train_sequences + 1

        # self.num_train_sequences = self.num_train_sequences + 1
        with open(dev_file) as f:
            for line in f:
                line = line.replace("\n", "").replace("\r", "")
                if not line.startswith("#"):
                    self.num_dev_examples = self.num_dev_examples + 1
                    if self.num_dev_examples % 50000 == 0:
                        print("Read", self.num_dev_examples, "train examples")
                    parts = line.split("\t")
                    if (len(parts) != 1):
                        if "-" not in parts[0] and "_" not in parts[6]:
                            word = parts[1].lower()
                            if word not in trainableWE:
                                trainableWE[word] = 1
                    else:
                        self.num_dev_sequences = self.num_dev_sequences + 1
        # self.num_dev_sequences = self.num_dev_sequences + 1

        # print "Reading word embeddings file"
        if False:
            with open(word_embeddings_file) as f:
                first_line = True
                for line in f:
                    line = line.replace("\n", "").replace("\r", "")
                    if first_line:
                        first_line = False
                    else:
                        self.num_embeddings = self.num_embeddings + 1
                        if self.num_embeddings % 10000 == 0:
                            print("Read", self.num_embeddings, "word embeddings")
                        parts = line.split(" ")
                        word = parts[0]
                        if word in trainableWE:
                            embeddings = [float(0)] * (len(parts) - 2)
                            self.word2int[word] = len(self.word2int)
                            for zz in range(len(parts) - 2):
                                embeddings[zz] = float(parts[zz + 1])
                            self.word_embeddings_size = len(parts) - 2
                            self.word2vec[word] = embeddings

        sys.stdout.write("Loaded " + str(len(self.word2vec)) + " trainable word embeddings of size " + str(
            self.word_embeddings_size) + "\n")
        sys.stdout.write("Train sequences " + str(self.num_train_sequences) + " with a total number of " + str(
            self.num_train_examples) + " examples\n")
        sys.stdout.write("Dev sequences " + str(self.num_dev_sequences) + " with a total number of " + str(
            self.num_dev_examples) + " examples\n")
        sys.stdout.write("Found " + str(len(self.xpos2int)) + " unique XPOS tags, " + str(
            len(self.upos2int)) + " unique UPOS tags and " + str(len(self.attrs2int)) + " unique attribute sets\n")
        self.label2int["NO_LABEL"] = len(self.label2int)
        sys.stdout.write("Found " + str(len(self.label2int)) + " unique labels\n")

        self.labels = [""] * len(self.label2int)
        for label in self.label2int:
            self.labels[self.label2int[label]] = label
        print
        self.labels
        # self.input_size = len(self.xpos2int) + len(self.upos2int) + self.word_embeddings_size
        self.output_size = len(self.label2int)
        self.aux_size = (len(self.upos2int) + len(self.xpos2int) + len(self.label2int)) * 2
        if use_morph:
            self.input_size = self.word_embeddings_size + 2 * embeddings_size
        else:
            self.input_size = embeddings_size#self.word_embeddings_size + embeddings_size

    def get_next_train_sequence(self):
        if (self.file_train is None):
            self.file_train = open(self.train_file)

        example = []
        example.append("0\t__root__\troot\troot\troot\troot\t-1\tNone")
        while True:
            line = self.file_train.readline()
            line = line.replace("\n", "").replace("\r", "")
            if (line == ""):
                if self.file_train.tell() == os.fstat(self.file_train.fileno()).st_size:
                    # print " reset"
                    self.file_train.close()
                    self.file_train = None

                break
            if not line.startswith("#"):
                example.append(line)
        return example

    def get_next_dev_sequence(self):
        if (self.file_dev is None):
            self.file_dev = open(self.dev_file)

        example = []
        example.append("0\t__root__\troot\troot\troot\troot\t-1\tNone")
        while True:
            line = self.file_dev.readline()
            line = line.replace("\n", "").replace("\r", "")
            if (line == ""):
                if self.file_dev.tell() == os.fstat(self.file_dev.fileno()).st_size:
                    # print " reset"
                    self.file_dev.close()
                    self.file_dev = None
                break
            if not line.startswith("#"):
                example.append(line)
        return example

    def get_x_t(self, parts, runtime):
        x = [float(0)] * self.input_size
        word = parts[1]
        xpos = parts[4]
        upos = parts[3]

        if upos in self.upos2int:
            x[self.upos2int[upos]] = 1

        ofs = len(self.upos2int)
        if xpos in self.xpos2int:
            x[ofs + self.xpos2int[xpos]] = 1

        ofs = len(self.xpos2int) + len(self.upos2int)
        if word in self.word2vec:
            emb = self.word2vec[word]
            for zz in range(self.word_embeddings_size):
                x[ofs + zz] = emb[zz]
        # print word+" "+upos+" "+xpos+" "+str(x)
        return x

    def get_x_stripped(self, parts, runtime):
        # x = [float(0)] * self.input_size
        word = parts[1]


        emb_char = dy.lookup(self.word_char_lookup, self.word2vec_character2int[word], update=False)
        return emb_char

    # def get_x_stripped(self, parts, runtime):
    #     # x = [float(0)] * self.input_size
    #     word = parts[1]
    #     if not runtime:
    #         p1 = random.random()
    #         p2 = random.random()
    #     else:
    #         p1 = 1
    #         p2 = 1
    #
    #     scale = 1
    #     if p1 < self.drop_prob:
    #         scale += 1
    #     if p2 < self.drop_prob:
    #         scale += 1
    #     scalScale = dy.scalarInput(scale)
    #
    #     if p1 > self.drop_prob:
    #         if word in self.word2int:
    #             index = self.word2int[word]
    #         elif "</s>" in self.word2int:
    #             index = self.word2int["</s>"]
    #         emb_word = dy.lookup(self.lookup_word, index, update=False) * scalScale
    #     else:
    #         emb_word = dy.vecInput(self.word_embeddings_size)
    #         emb_word.set(np.zeros((self.word_embeddings_size)))
    #
    #     if p2 > self.drop_prob:
    #         emb_char = dy.lookup(self.word_char_lookup, self.word2vec_character2int[word], update=False) * scalScale
    #     else:
    #         emb_char = dy.vecInput(self.embeddings_size)
    #         emb_char.set(np.zeros((self.embeddings_size)))
    #
    #     return dy.concatenate([emb_word, emb_char])

    def get_x(self, parts, runtime):
        x = [float(0)] * self.input_size
        word = parts[1]
        xpos = parts[4]
        upos = parts[3]
        attrs = parts[5]

        if not runtime:
            p1 = random.random()
            p2 = random.random()
        else:
            p1 = 1
            p2 = 1

        scale = 1.0
        if p1 < self.drop_prob:
            scale = scale + 1.0
        if p2 < self.drop_prob:
            scale = scale + 1.0
        scalScale = dy.scalarInput(scale)
        if p2 > self.drop_prob and upos in self.upos2int:
            emb_upos = dy.lookup(self.lookup_upos, self.upos2int[upos]) * scalScale
        else:
            emb_upos = dy.vecInput(self.embeddings_size)
            emb_upos.set(np.zeros((self.embeddings_size)))

        if p2 > self.drop_prob and xpos in self.xpos2int:
            emb_xpos = dy.lookup(self.lookup_xpos, self.xpos2int[xpos]) * scalScale
        else:
            emb_xpos = dy.vecInput(self.embeddings_size)
            emb_xpos.set(np.zeros((self.embeddings_size)))
        if p1 > self.drop_prob:
            if word in self.word2int:
                index = self.word2int[word]
            elif "</s>" in self.word2int:
                index = self.word2int["</s>"]
            emb_word = dy.lookup(self.lookup_word, index, update=False) * scalScale
        else:
            emb_word = dy.vecInput(self.word_embeddings_size)
            emb_word.set(np.zeros((self.word_embeddings_size)))

        return dy.concatenate([emb_word, emb_xpos, emb_upos])

    def encode_input(self, example, runtime, use_morph):
        x = []
        aux_size = (len(self.label2int) + len(self.upos2int) + len(self.xpos2int)) * 2
        y = np.zeros((len(example), aux_size))
        for i in range(len(example)):
            parts = example[i].split("\t")
            if use_morph:
                xx = self.get_x(parts, runtime)
            else:
                xx = self.get_x_stripped(parts, runtime)
            x.append(xx)

        return x

    def encode_output(self, item1, item2):
        parts1 = item1.split("\t")
        parts2 = item2.split("\t")
        to = parts1[6]
        # label = parts1[7]
        id_dst = parts2[0]
        yy = [float(0)] * 2
        if to == id_dst:
            yy[0] = 1  # yy[self.label2int[label]] = 1
        else:
            yy[1] = 1  # yy[self.label2int["NO_LABEL"]] = 1
        return yy

    def store_embeddings(self, path):
        print
        "Storing labels and features"
        with open(path, "w") as f:
            f.write("UPOS " + str(len(self.upos2vec)) + "\n")
            for s in self.upos2vec:
                f.write(s + " " + self.vec2string(self.upos2vec[s]) + "\n")
            f.write("XPOS " + str(len(self.xpos2vec)) + "\n")
            for s in self.xpos2vec:
                f.write(s + " " + self.vec2string(self.xpos2vec[s]) + "\n")

    def vec2string(self, vec):
        s = str(vec[0])
        for zz in range(1, len(vec)):
            s += " " + str(vec[zz])
        return s

    def store_lookups(self, path):
        f = open(path, "w")
        f.write("UPOS " + str(len(self.upos2int)) + "\n")
        for s in self.upos2int:
            f.write(s + " " + str(self.upos2int[s]) + "\n")
        f.write("XPOS " + str(len(self.xpos2int)) + "\n")
        for s in self.xpos2int:
            f.write(s + " " + str(self.xpos2int[s]) + "\n")
        f.write("ATTRS " + str(len(self.attrs2int)) + "\n")
        for s in self.attrs2int:
            f.write(s + " " + str(self.attrs2int[s]) + "\n")

        f.write("LABELS " + str(len(self.label2int)) + "\n")
        for s in self.label2int:
            f.write(s + " " + str(self.label2int[s]) + "\n")
        f.write("WORDS " + str(len(self.word2int)) + "\n")
        for s in self.word2int:
            f.write(s + " " + str(self.word2int[s]) + "\n")

        if self.use_morph == False:
            f.write("WORD_CHARS " + str(len(self.word2vec_character2int)) + "\n")
        f.close()

    def restore_lookups(self, path):
        f = open(path)
        parts = f.readline().replace("\n", "").split(" ")
        self.upos2int = {}
        self.upos_list=[]
        for _ in xrange(int(parts[1])):
            self.upos_list.append('')
        for zz in range(int(parts[1])):
            parts = f.readline().replace("\n", "").split(" ")
            self.upos2int[parts[0]] = int(parts[1])
            self.upos_list[int(parts[1])]=parts[0]
        print self.upos_list

        parts = f.readline().replace("\n", "").split(" ")
        self.xpos2int = {}
        self.xpos_list=[]
        for _ in xrange(int(parts[1])):
            self.xpos_list.append('')
        for zz in range(int(parts[1])):
            parts = f.readline().replace("\n", "").split(" ")
            self.xpos2int[parts[0]] = int(parts[1])
            self.xpos_list[int(parts[1])]=parts[0]

        parts = f.readline().replace("\n", "").split(" ")
        self.attrs2int = {}
        self.attrs_list=[]
        for _ in xrange(int(parts[1])):
            self.attrs_list.append('')
        for zz in range(int(parts[1])):
            parts = f.readline().replace("\n", "").split(" ")
            self.attrs2int[parts[0]] = int(parts[1])
            self.attrs_list[int(parts[1])]=parts[0]


        parts = f.readline().replace("\n", "").split(" ")
        self.label2int = {}

        for zz in range(int(parts[1])):
            parts = f.readline().replace("\n", "").split(" ")
            self.label2int[parts[0]] = int(parts[1])
        self.labels = [""] * len(self.label2int)
        for label in self.label2int:
            self.labels[self.label2int[label]] = label
        parts = f.readline().replace("\n", "").split(" ")

        self.word2intTmp = self.word2int
        self.word2vecTmp = self.word2vec
        self.word2int = {}
        self.word2vec = {}
        for zz in range(int(parts[1])):
            parts = f.readline().replace("\n", "").split(" ")
            self.word2int[parts[0]] = int(parts[1])
            self.word2vec[parts[0]] = [0] * self.word_embeddings_size
        if self.use_morph == False:
            parts = f.readline().replace("\n", "").split(" ")
            self.word2vec_character2int = {}
            self.word2vec_character = {}
            for zz in range(int(parts[1])):
                self.word2vec_character2int[str(zz)] = zz
                self.word2vec_character[str(zz)] = [float(0)] * self.embeddings_size
        f.close()
