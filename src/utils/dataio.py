#-*-coding:UTF-8
'''
Created on 2016-8-3-11:15:25
author: Gary-W
'''
import os
import pickle
import shutil
import numpy as np
import json

# get all dir' abs path in given dir
def get_dirlist(root_dir, includes=[], excludes=[]):
    """
    sample:
    get_dirlist(r'../daytime_adaboost/pos')
    """
    p = []
    fns = os.listdir(root_dir)
    for fn in fns:
        okey = 0
        if includes or excludes:
            for inc in includes:
                if inc in fn: okey = 1; break
            for exc in excludes:
                if exc in fn: okey = 0; break
        else:
            okey = 1
        if okey:
            fpath = os.path.join(root_dir, fn)
            if os.path.isdir(fpath):
                p.append(fpath)
    return p

# get all files' abs path in given dir, subfix is variable len parameter
def get_filelist(root_dir, *subfixs):
    """
    sample:
    get_filelist(r'../daytime_adaboost/pos',".png",".jpg")
    """
    p = []
    for subfix in subfixs:
        p.extend([os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.endswith(subfix)])
    return p

# get all files' name.* in given dir, subfix is variable len parameter
def get_filenamelist(root_dir, *subfixs):
    p = []
    for subfix in subfixs:
        p.extend([f for f in os.listdir(root_dir) if f.endswith(subfix)])
    return p

# get all files' abs path in given dir, prefix is variable len parameter
def get_prefilelist(root_dir, *prefixs):
    p = []
    for prefixs in prefixs:
        p.extend([os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.startswith(prefixs)])
    return p

# get all files' abs path in given tree, subfix is variable len parameter
def get_walkfilelist(root_dir, *subfixs):
    fullpath = []       # absolute path in the tree
    relate_path = []      # relative path in the tree
    len_root = len(root_dir)
    for root, _dirs, files in os.walk(root_dir): 
        for filespath in files:
            for subfix in subfixs:
                if filespath.endswith(subfix):
                    p = os.path.join(root,filespath)
                    fullpath.append(p)
                    relate_path.append(p[len_root+1:])    # backspace an "\"
    return fullpath, relate_path

# load json file as dict
def load_json(path):
    """
    note: json str("..."),'/', never end with ","
    """
    js_table = None
    try:
        with open(path, 'r') as fr:
            js_table = json.load(fr)
            js_table["json_path"] = path
    except IOError as ioerr:
        print("IO Error:"+str(ioerr)+"in:\n"+path)
    return js_table

# pickle files
def store_pickle(path, obj):
    try:
        with open(path, 'wb') as fw:
            pickle.dump(obj, fw)
    except IOError as ioerr:    print("IO Error:"+str(ioerr)+"in:\n"+path)

def load_pickle(path):
    try:
        with open(path, 'rb') as fr:
            obj = pickle.load(fr)
            return obj
    except IOError as ioerr:    print("IO Error:"+str(ioerr)+"in:\n"+path)

def save_npy(path, dict_data):
    np.save(path, dict_data)

def load_npy(path):
    # this can be used in py3.5
    data_dict = np.load(path, encoding="latin1").item()
    return data_dict

# create dir of training set based on src_dir & dst_dir
def create_dirs(src_dir, dst_dir):
    """
    src_dir include:
    class1, class2, ... dir etc
    dst_dir include:
    train, val, train.txt, val.txt
    """
    label_names = os.listdir(src_dir)
    for label in label_names:
        train_dir = os.path.join(dst_dir,"train", label)
        val_dir = os.path.join(dst_dir,"val", label)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
    return label_names


# copy src dir-tree 2 dst dir-tree, only include the files with given subfix 
def copy_n_sel(src, dst, *ig_subfixs):
    if not os.path.exists(dst):
        os.makedirs(dst)
    _, sel_corr_paths = get_walkfilelist(src, ig_subfixs)
    dst_list = []
    for sel_p in sel_corr_paths:
        dst_path = os.path.join(dst, sel_p)
        src_path = os.path.join(src, sel_p)
        parent_dir = os.path.split(dst_path)[0]
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        print("copying ",dst_path)
        shutil.copyfile(src_path,dst_path)
        dst_list.append(dst_path)
    return dst_list


# find a Drive including the target dir in root directory
def getTargetDisk(tar_file="Location_PoseNet_Code_Dataset"):
    candidate = ["E:","F:","G:","H:","I:"]
    for d in candidate:
        if os.path.isdir(d):
            tmpath = os.path.join(d, tar_file)
            if os.path.exists(tmpath):
                return d
    print("no valid disk")
    return ""

def get_filesize(fpath):
    """
    return the size(string-type) of a file
    """
    
    def formatSize(bytes_number):
        # turn bytes number into kb\m\g unit
        try:
            bytes_number = float(bytes_number)
            kb = bytes_number / 1024
        except:
            print("invalid input format")
            return("Error")
        if kb >= 1024:
            M = kb / 1024
            if M >= 1024:
                G = M / 1024
                return("%fG" % (G))
            else:
                return("%fM" % (M))
        else:
            return("%fkb" % (kb))
        
    return formatSize(os.path.getsize(fpath))

class MagaDict:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def __getitem__(self, key):
        try:
            print(key,":",self.config_dict[key])
            return self.config_dict[key]
        except:
            return None
        
    def __setitem__(self,key,value):
        print('"'+key+'"',"<==",value)
        self.config_dict[key] = value
    
    def savefile(self, file):
        with open(file,"a") as fa:
            for key in self.config_dict:
                fa.write(str(key)+":"+str(self.config_dict[key])+"\n")
        
            
def restore_logs_to_file(dst_file, src_strlog):
    with open(dst_file,"w") as fw:
        for log in src_strlog:
            fw.write(log+"\n")

def restore_logs_from_file(src_file):
    src_strlog = []
    with open(src_file,"w") as fw:
        for log in fw:
            src_strlog.append(log.strip("\n"))
    return src_strlog



if __name__=="__main__":
    pass


