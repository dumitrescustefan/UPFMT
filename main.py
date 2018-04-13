import sys 
import os
from os import listdir
from os.path import isfile, join, split
import subprocess
from shutil import copyfile
from parser.neural_parser import test

#python2 main.py --input=/work/_d/in --output=/work/_d/out --param:language=ro

def convert_xmi2txt (xmi_file, txt_file):
    contents = []   
    with open(xmi_file,'r') as fr:
        contents = fr.readlines()       
    for line in contents:
        if "sofaString=" in line:
            text = line[line.index("sofaString=")+12:line.rfind("\" />")]                
    with open(txt_file,'w') as fr:
        fr.write(text)


print("Entrypoint in docker container:")
print(sys.argv[1:])
input=""
output=""
language=""
params={}

# read parameters
for param in sys.argv[1:]:    
    parts = param.split("=")
    if len(parts)!=2:
        print("Error parsing parameter: "+param)
        sys.exit(10)
    if parts[0]=="--input":
        input = parts[1]
        continue
    if parts[0]=="--output":
        output = parts[1]
        continue
    if "--param:" in parts[0]: #--param:language=en
        params[parts[0].split(":")[1]] = parts[1]        
        if "language" in params:
            language = params["language"]
    
print("Docker input folder: "+input)
print("Docker output folder: "+output)
print("Language parameter:  "+language)
print("Other parameters: "+str(params))
root_folder = os.path.dirname(os.path.realpath(__file__))
print("Local path is: "+root_folder)
print("_"*80)

# ############## INPUT ###########################################
# read all input files
input_files_xmi = [os.path.join(input,f) for f in listdir(input) if isfile(os.path.join(input, f)) and ".xmi" in f]
input_files_txt = [os.path.join(input,f) for f in listdir(input) if isfile(os.path.join(input, f)) and ".txt" in f]
if len(input_files_xmi) + len(input_files_txt) == 0:
    print(" No input .xmi or .txt files found!")
    sys.exit(0)


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
    command="java -Xmx2g -jar "+root_folder+"/tools/UDTokenizer.jar "+root_folder+"/tools/models/"+language+" "+input_file+" "+root_folder+"/temporary.conll"
    print("\n\t\t Running tokenizer : "+command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()    
    
    copyfile(root_folder+"/temporary.conll", output+"/temporary.conll")
    
    # run all other tools
    test(root_folder+"/models/"+language, root_folder+"/temporary.conll", output_file)
    
    # convert conllu to xmi
    if os.path.isfile(output_file.replace(".conllu",".xmi")): # cleanup of existing .xmi file is necessary so groovy script dosen't fail
        print("\t\t Removing existing file : "+output_file.replace(".conllu",".xmi"))
        os.remove(output_file.replace(".conllu",".xmi")) 
        os.remove(os.path.join(output,"TypeSystem.xml"))
        
    #args[0] <- input path
    #args[1] <- input conllu file
    #args[2] <- output path to write xmi file
    _, output_file_name = split(output_file)
    command="groovy /conllu2xmi.groovy "+output+" "+output_file_name+" "+output
    print("\n\t\t Running conllu2xmi : "+command)    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
