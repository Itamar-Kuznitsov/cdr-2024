from enum import unique
from operator import truth
import yara
from pdf.PDF_Disarm_v2 import *
from NNPyTorchModel.ModelPaeser import *
from hwp.HWP_disarm import HwpFile
from os import listdir
from os.path import isfile, join
import time

def fetch_yara_rules(file_type: str) -> list:
    """
    fetch all yara rules from YARA-Rules directory by spesific file type identifier
    return list of rules
    """
    rules = []
    directory_path = f"./YARA-Rules/{file_type}"
    print(directory_path)
    file_names = [file for file in os.listdir(directory_path) if file.endswith(".yar")]
    print(file_names)

    for file_name in file_names:
        rule_path = directory_path + "/" + file_name
        rule = yara.compile(filepath=rule_path) 
        rules.append(rule)

    return rules

def get_yara_rules(file_type: str) -> list:
    """
    for testing purpose we will choose the rules not from db but 'hard coded'
    get a file type and returns  list of rules as given from YARA
    """
    # TODO(IK): search hwp files
    if file_type == 'hwp':
        return fetch_yara_rules('hwp')
    
    elif file_type == 'pdf':
        return fetch_yara_rules('pdf')
    
    elif file_type == 'pyth':
        return []
    
    else:
        raise TypeError(file_type)
    return []

def get_cdr_rules(file_type: str):
    """
    for testing purpose we will choose the rules not from db but 'hard coded'
    get a file type and returns  list of rules as given from YARA
    """
    if file_type == 'pdf':
        cdr_rules_pdf = {"root": ['r', [b'/OpenAction', b'/JavaScript', b'/JS', b'/AcroForm', b'/JS', b'/AA',b'/URI']],
            "/JavaScript" : ['ra'],
            "/Font": ['c', [b'<</Type /Font\n/BaseFont /Helvetica\n/Subtype /Type1\n>>']],
            "/DecodeParms": ['r', [b'/Colors']],
            '/Metadata': ['r' ,[b'asd']],
            "/Page":['r', [b'/OpenAction', b'/JavaScript', b'/JS', b'/AcroForm', b'/JS', b'/AA',b'/URI']],
            "default": ['ra']}
        return cdr_rules_pdf
    elif file_type == 'hwp':
        cdr_rules_hwp = {}
        
    elif file_type == 'pyth':
        cdr_rules_nn = {"weights": ['c', ['10']]}#other option is to use n-random ['c', ['10']]
        return cdr_rules_nn

    else:
        raise TypeError(file_type)

def get_yara_offset(file_name: str, rules):
    """
    extract the detected rules offsets return the offsets and the rules detected
    """
    offset_list = []
    rules_names = []
    for r in rules:
        try:
            rule_d = r.match(file_name)
            rules_names.append(rule_d)
            for i in rule_d:
                for j in i.strings:
                    offset_list.append(j[0])
        except:
            offset_list.append([1])
            rules_names.append(["there is a problem OR file type does not suppurt match(for now only pyth type)"])

    return offset_list , rules_names

def disassemble_file(file_name: str, f_type):
    """
    call file 'disassemble' by type and return the object
    """
    if f_type == 'pdf':
        return PdfFile(file_name)
    elif f_type == 'hwp':
        return HwpFile(file_name)
    elif f_type == 'pyth':
        return ModelParser(file_name)
    else:
        raise TypeError(f_type)

def find_object_detected(offset_list, file_obj):
    """
    gets the objects detected by the rules (there id but id as reletive(real id like pdf or imagenry one))
    """
    id_list = []
    for i in offset_list:
        
        x = file_obj.get_id_by_offset(i) #for every file we need this function
        id_list.append(x)
    return id_list

def run_cdr_rules(file_obj, cdr_rules, id_list):
    """
    for each object run the cdr rules for him 
    the remove object by id checks for rules for the objects if not found removes it
    """
    for i in id_list:
        if i != 0:
            file_obj.remove_obj_by_id(i, cdr_rules) #for every file we need this function
    return file_obj



def CDR_a_file(dir_name, file_name):
    """
    get a file name and dir 
    @p is dir name 
    @n is file name 
    does the CDR steps: 
    get yara rules ->
    get all offsets of the yara rules deteted ->
    dissassmble the file by his type (need change for this code) ->
    get all the objects id that was detected (id is reletive for the file may not realy be an id) ->
    run cdr rules (need to be changed because now its run always pdf rules) ->
    dump the file ->
    """
    full_name = dir_name+ "/" +file_name
    #note this part shood be fixed by getting the abssalute file type (maybe using mimetypes)
    file_type = file_name.split(".")[1]
    print(file_type)
    #fixed
    cdr_rules = get_cdr_rules(file_type)
    #fixed
    yara_list = get_yara_rules(file_type)
    #fine
    offset_list, rules_detected = get_yara_offset(full_name, yara_list)
    print("rules detected - ", rules_detected)
    #fine
    diss_file = disassemble_file(full_name, file_type)
    #every parser shood have get_id_by_offset function to use this
    id_list = find_object_detected(offset_list, diss_file)
    #every parser shood have remove_obj_by_id function to use this
    diss_file = run_cdr_rules(diss_file, cdr_rules, id_list)#use by type
    #can maybe change the output dir to be given and to a dir with a timestemp so no overiding will happen
    diss_file.dump_file("OUTPUT",full_name[len(mypath):]) #for every file we need this function


#
mypath = "./FILES"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in onlyfiles:
    print("================================================")
    print(i)
    all_p = mypath + '/' + i
    print(all_p)
    try:
        start = time.time()
        CDR_a_file(mypath,i)
        end = time.time()
        with open(f"./LOGS/{i}.log", 'w+') as f:
                    print("time- ", end-start)
                    f.write(str(end-start))

    except Exception as e:
        print("Failed with",e)