import sys 
import os
from os import listdir
from os.path import isfile, join, split
import subprocess
from parser.neural_parser import test

#python2 main.py --input=/work/_d/in --output=/work/_d/out --param:language=ro

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
print("\nOther parameters:")
print(params)
root_folder = os.path.dirname(os.path.realpath(__file__))
print("Local path is: "+root_folder)
print("_"*80)
# read all input files

input_files = [os.path.join(input,f) for f in listdir(input) if isfile(join(input, f)) and ".xmi" in f]
if len(input_files) == 0:
    print(" No input .xmi files found!")
    sys.exit(0)

 
for input_file in input_files:
    _, input_file_name = split(input_file)
    output_file = os.path.join(output,input_file_name.replace(".xmi",".conllu"))
    print("\n\tInput file : "+input_file)
    print("\tOutput file: "+output_file)
    
    #convert from xmi to txt
    txt_filepath = os.path.join(output,input_file_name.replace(".xmi",".txt"))
    contents = ""    
    with open(input_file,'r') as fr:
        contents = fr.readlines()   
    for line in contents:
        if "sofaString=" in line:
            text = line[line.index("sofaString=")+12:line.rfind("\" />")]
            
            print("\n\t\t Converted xmi to :"+input_file.replace(".xmi",".txt"))
            with open(txt_filepath,'w') as fr:
                fr.write(text)
    
    input_file = txt_filepath
    
    #tokenization --> parsing
    #    print "Usage: main.py <language code> <input raw text> <output conll>"  
    command="java -Xmx2g -jar "+root_folder+"/tools/UDTokenizer.jar "+root_folder+"/tools/models/"+language+" "+input_file+" "+root_folder+"/tmp/temporary.conll"
    print("\n\t\t Running : "+command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()    
    test(root_folder+"/models/"+language, root_folder+"/tmp/temporary.conll", output_file)
    
    # convert conllu to xmi
    #args[0] <- input path
    #args[1] <- input conllu file
    #args[2] <- output path to write xmi file
    _, output_file_name = split(output_file)
    command="groovy /conllu2xmi.groovy "+output+" "+output_file_name+" "+output
    print("\n\t\t Running : "+command)    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
