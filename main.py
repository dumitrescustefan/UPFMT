import sys 
import os
from os import listdir
from os.path import isfile, join, split
import subprocess
from shutil import copyfile
from parser.neural_parser import test
import lxml.html
import codecs
import re

#python2 main.py --input=/work/_d/in --output=/work/_d/out --param:language=ro

def convert_xmi2txt (xmi_file, txt_file):
    contents = []   
    with open(xmi_file,'r') as fr:
        contents = fr.readlines()       
    for line in contents:
        if "sofaString=" in line:
            text = line[line.index("sofaString=")+12:]
            i1 = text.find("\" />")
            i2 = text.find("\"/>")
            if i1!=-1 and i2!=-1:
                i = min(i1,i2)
            if i1!=-1 and i2==-1:
                i = i1
            if i1==-1 and i2!=-1:
                i = i2
            if i1==-1 and i1==-1:
                i = len(text) # this should not happen            
            text = text[:i].strip()
    
    if text.endswith("\"/>"):
        text = text[:-3]
    if text.endswith("\" />"):
        text = text[:-4]        
     
    # clean up text ...
    text = lxml.html.fromstring(text).text
    text = re.sub(' +',' ',text)
    
    #with open(txt_file,'w') as fr:
    with codecs.open(txt_file,'w',encoding='utf8') as fr:
        fr.write(text)

def mem_protection (input_file, output_file):
    contents = []   
    output = []
    max = 150
    with codecs.open(input_file,'r',encoding='utf8') as fr:
            contents = fr.readlines()         
        
    counter = 0
    for line in contents:
        l = line.strip()
        if l=="":
            counter = 0 
            output.append("\n")
        else:
            counter+=1
            index = l.find("\t")
            if counter > max:
                counter = 1
                output.append("\n")
            output.append(str(counter)+l[index:]+"\n")
    
    output.append("\n")      
    with codecs.open(output_file,'w',encoding='utf8') as fr:
        for l in output:
            fr.write(l)
        

print("Entrypoint in docker container:")
print(sys.argv[1:])
input=""
output=""
language="en" #default language
params={}

# read input 
for i in range(1,len(sys.argv)-1):
    param = sys.argv[i]
    if param == "--input":
        input = sys.argv[i+1]
    if param == "--output":
        output = sys.argv[i+1]

for param in sys.argv[1:]:    
    if "=" not in param:
        continue;    
    parts = param.split("=")
    if len(parts)!=2:
        print("Error parsing parameter: "+param)
        sys.exit(10)
    if "--param:" in parts[0]: #--param:language=en
        params[parts[0].split(":")[1]] = parts[1]        
        if "language" in params:
            language = params["language"]

if input.strip()=="":
    print("ERROR: Please provide an input folder!")
    sys.exit(1)
if output.strip()=="":
    print("ERROR: Please provide an output folder!")
    sys.exit(2)
if input==output:
    print("ERROR: Input folder must not be the same as the output folder!")
    sys.exit(3)
            
print("Docker input folder: "+input)
print("Docker output folder: "+output)
# check output folders exist
if not os.path.exists(output):
    os.makedirs(output)
    print("\t folder does not exist, creating it..")
    if not os.path.exists(output):
        print("\t folder creation failed! ["+output+"] does not exist")
        sys.exit(11)        
print("Language parameter:  "+language)
# check that language model exists
if not os.path.exists(os.path.join(os.sep, "UPFMT","models",language)):
    print("\t ERROR, model for language ["+language+"] does not exist, exiting...")
    sys.exit(12)        

print("Parameter dictionary: "+str(params))
root_folder = os.path.dirname(os.path.realpath(__file__))
print("Local path is: "+root_folder)
print("_"*80)

# ############## INPUT ###########################################
# read all input files
input_files_xmi = [os.path.join(input,f) for f in listdir(input) if isfile(os.path.join(input, f)) and ".xmi" in f]
input_files_txt = [os.path.join(input,f) for f in listdir(input) if isfile(os.path.join(input, f)) and ".txt" in f]
if len(input_files_xmi) + len(input_files_txt) == 0:
    print(" No input .xmi or .txt files found!")
    sys.exit(4)


# 1. convert xmi to txt
print("Step 1a. Converting existing .xmi files to .txt ...")
for input_file_xmi in input_files_xmi:
    _, filename = split(input_file_xmi)
    output_file_txt = os.path.join(output, filename.replace(".xmi",".txt"))
    print ("\t Converting ["+filename+"] to ["+output_file_txt+"] ...")
    convert_xmi2txt(input_file_xmi,output_file_txt)

# 2. copy txt files to acsaxcas
print("Step 1b. Copying existing .txt files unchanged ...")
for input_file_txt in input_files_txt:
    _, filename = split(input_file_txt)
    output_file_txt = os.path.join(output, filename)
    print ("\t Copying ["+filename+"] to ["+output_file_txt+"] ...")
    copyfile(input_file_txt, output_file_txt)

    
# ############## PROCESS #########################################   
print("Step 2. Processing files ...")
input_files = [os.path.join(output,f) for f in listdir(output) if isfile(os.path.join(output, f)) and ".txt" in f]
for input_file in input_files:
    _, input_file_name = split(input_file)
    output_file = os.path.join(output,input_file_name.replace(".txt",".conllu"))
    
    print("\n\tInput file : "+input_file)
    print("\tOutput file: "+output_file)    
        
    #tokenization --> parsing
    #    print "Usage: main.py <language code> <input raw text> <output conll>"  
    command="java -Xmx1g -jar "+root_folder+"/tools/UDTokenizer.jar "+root_folder+"/tools/models/"+language+" "+input_file+" "+root_folder+"/temporary.conllu"
    print("\n\t\t Running tokenizer : "+command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()    
    
    # debug intermediary step
    copyfile(root_folder+"/temporary.conllu", os.path.join(output,"temporary.conllu"))
    
    # mem protection, force line split
    mem_protection(root_folder+"/temporary.conllu", root_folder+"/temporary-mem.conllu")
    
    # debug intermediary step
    copyfile(root_folder+"/temporary-mem.conllu", os.path.join(output,"temporary-mem.conllu"))    
    
    # run all other tools
    test(root_folder+"/models/"+language, root_folder+"/temporary-mem.conllu", output_file)
    
    # convert conllu to xmi
    if os.path.isfile(os.path.join(output,"TypeSystem.xml")):
        os.remove(os.path.join(output,"TypeSystem.xml"))
    if os.path.isfile(output_file.replace(".conllu",".xmi")): # cleanup of existing .xmi file is necessary so groovy script dosen't fail (its overwrite defaults to false)
        print("\t\t Removing existing file : "+output_file.replace(".conllu",".xmi")) 
        os.remove(output_file.replace(".conllu",".xmi")) 
        
    #args[0] <- input path
    #args[1] <- input conllu file
    #args[2] <- output path to write xmi file
    _, output_file_name = split(output_file)
    command="groovy /conllu2xmi.groovy "+output+" "+output_file_name+" "+output
    print("\n\t\t Running conllu2xmi : "+command)    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
print("DONE.")