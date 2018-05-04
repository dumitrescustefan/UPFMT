[![Documentation Status](http://readthedocs.org/projects/upfmt/badge/?version=latest)](http://upfmt.readthedocs.io/en/latest/?badge=latest)
  
# UPFMT
Unified Processing Framework for raw Multilingual Text

UPFMT is a lightweight and easy to use tool for converting raw text into Universal Dependencies (aka CONLLU format) that support all major languages from the UD corpus (http://universaldependencies.org/)

Those in a hurry to get started should first go through the Prerequisites section and then directly to the QuickStart guide. However, building and tuning your own system will take time and effort - we provide full technical details as-well as a guide to train our system on your own data.

## Prerequisites
- Python 2.7
- JAVA > 1.8
- DyNET (https://github.com/clab/dynet)
- Pretrained models (included) or data from the UD corpus (http://universaldependencies.org/)

Python 2.7 is included in major Linux distributions and is easy to install for Windows or OSX-based systems. If your OS does not include Python 2.7, check https://wiki.python.org/moin/BeginnersGuide/Download for installation instructions.
Also, JAVA/OpenJDK should be easily installable via major package manegement systems such as ```yum``` and ```apt``` or by downloading the binary distribution from Oracle (https://www.oracle.com/java/index.html)

Pretrained models are already included in the standard repository and dynet install will be covered in the quick-start quide.

## Quick start guide

First, make sure ```pip``` is installed with your Python 2.7 distribution. If not:
for Debian/Ubuntu
```sh
$ sudo apt-get install python-pip
```
or for Redhat/CentOS
```sh
$ yum install python-pip
```
Next, install DyNET:
```sh
$ pip install git+https://github.com/clab/dynet#egg=dynet
```
Next, get UPFMT by downloading the ZIP arhcive or by cloning this REPO using GIT:
```sh
$ cd ~
$ git clone https://github.com/dumitrescustefan/UPFMT.git
```
You can now do a dry run of the system to see if everything is set up correctly. In the folder where you cloned or downloaded and extracted this repo, type:
```sh
$ cd UPFMT
$ mkdir test; mkdir test/in mkdir test/out
$ echo "This is a simple test." > test/in/input.txt
$ python2 main.py --input=test/in --output=test/out --param:language=en
```

If everything worked fine, after the last command you should have a file with your results in the ```test/out``` folder:
```sh
$ cat test/out/input.conllu
1	This	this	PRON	DT	Number=Sing|PronType=Dem	0	-	_	_
2	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	-	_	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	0	-	_	_
4	simple	simple	ADJ	JJ	Degree=Pos	0	-	_	_
5	test	test	NOUN	NN	Number=Sing	0	-	_	SpaceAfter=No
6	.	.	PUNCT	.	_	0	-	_	_
```

### Advanced
The instructions above cover the one-liner installation of DyNET. It is sufficient if you only want to run the software and not train your own models. However, good speedups both in runtime and training time are obtained by building your own DyNET from source. As such, we recommend you follow the instructions at https://github.com/clab/dynet and build DyNET with support for Intel's Math Kernel Library (MKL) (https://software.intel.com/en-us/mkl) and NVIDIA CUDA (https://developer.nvidia.com/cuda-zone) and CuDNN support if you own a high-end graphics card (otherwise, MKL will be fine).

## Training your own models

For those interested in training their own models we have prepared a guide, describing the architecture of our neural approach and the steps necessary for building new models.

### System architecture
Our system is based on bidirectional Long-Short-Term Memory (LSTM) networks that are applied over the input sequence in an attempt to learn long-range dependencies between words inside an utterance. The input features are composed of lexicalized features (character embeddings) and hollistic word embeddings, which are task-oriented (optimized during training). The lexicalized features are obtained by applying a unidirectional LSTM over character embeddings and using the final state. During the early training stage, these character-level embeddings are pretrained by optimizing the system to output a word's part of speech based on its composing characters.

Parsing and part-of-speech tagging are carried out at the same time, just like standard multitask learning. During optimization we store the model which outputs the best Unlabeled Attachement Score (UAS) over the development set.

To train a new model you need to have a training and development set in CONLLU format. You start by downloading the UD corpus from the UD website:
```sh
$ wget http://ufal.mff.cuni.cz/~zeman/soubory/release-2.2-st-train-dev-data.zip
$ unzip release-2.2-st-train-dev-data.zip
```
Once the models are downloaded, you can train a new model by running the command below (applies to English):
```sh
$ python parser/neural_parser.py --train ud-treebanks/UD_English/en-ud-train.conllu ud-treebanks/UD_English/en-ud-train.conllu models/en/parser 
```
The ``-train`` command has three parameters: train file (in conllu format), dev file (in conllu format) and model storage location. Please note that the model storage location ends not with a folder, but with the prefix of all the files that will be created. In the example above, everything will be stored in `models/en/` and several files will be generated, like : ``parser.last``, ``parser.bestUAS``,  ``parser-character.network``, etc. 

At run-time, the same model path will need to be given.

Any language can be used as a corpus to build a new model as long as the corpus is in conllu format and we have created a train-dev pair. Not having a dev file will also work (wont't break the training), but stopping the training will be done by evaluation the parsing performance over the train set, a "trick" that will most likely lead to overfitting the training data and implicit worse performance results on new examples; so, we highly recommend having a dev corpus set aside when training.
