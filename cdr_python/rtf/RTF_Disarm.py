import os
import sys

class RTFNode:
    s_offset = None
    e_offset = None
    raw_string = None
    n_type = None
    childs = None
    id = None
    parent = None
    level = 0
    def __init__(self, start:int, end:int, data:str, type_of:str, level:int) -> None:
        self.s_offset = start
        self.e_offset = end
        self.raw_string = data
        self.n_type = type_of
        self.id = start
        self.level = level
        self.childs = []

    def add_child(self, child):
        child.parent = self
        self.childs.append(child)

    def remove_child_by_id(self,id:int):
        id_to_remove = None
        for i in len(self.childs):
            if self.childs[i].id == id:
                id_to_remove = i
        if id_to_remove is None:
            return
        self.childs.pop(i)

    def __repr__(self,end = False) -> str:
        ret = ""
        for i in range(0,self.level):
            ret += ""
            if i == self.level - 1 and end:
                ret += "\-->"
            elif i == self.level - 1:
                ret += "|-->"
            else:
                ret += "|       "
        ret+= str(self.id)+","+str(self.s_offset)+" - "+ str(self.e_offset)+"\n"
        for i,child in enumerate(self.childs):
            if i == len(self.childs)-1:
                ret += child.__repr__(True)
            else:
                ret += child.__repr__()
        return ret

class RTFTree:
    """
    this class represnts the rtf file parts using a tree 
    """
    size = 0
    raw_data = ''
    head_node = None
    def __init__(self, raw_data: str) -> None:
        self.raw_data = raw_data
        self.head_node , _= self.create_tree(raw_data,-1, 0)


    def create_tree(self, raw_data: str, prev_offset: int , level: int):
        start_offset = prev_offset + 1
        end_offset = prev_offset + 1
        str_data = ''
        type_data = ''
        childs = []

        count_left = 0

        runner = 0
        while runner < len(raw_data):

            if raw_data[runner] == '{' and count_left >= 1:
                count_left+=1
                c, e = self.create_tree(raw_data[runner:], start_offset+runner-1, level+1)
                runner += e
                #print(raw_data[runner], level)
                childs.append(c)
            if raw_data[runner] == '{':
                    count_left+=1
            elif raw_data[runner] == '}':
                count_left-=1
            

            if count_left == 0:
                break 

            runner+=1
        if count_left != 0:
                raise "bad structure" 
        
        end_offset = start_offset + runner +1
        str_data = self.raw_data[start_offset:end_offset]
        type_data = 'cool'
        node = RTFNode(start_offset,end_offset,str_data,type_data, level)
        node.id = self.size
        self.size +=1 
        for i in childs:
            node.add_child(i)
        
        return node, runner

    def __repr__(self) -> str:
        return self.head_node.__repr__()

class RTF:
    tree = None
    def __init__(self, filename: str) -> None:
        file_data = ''
        with open(filename, "r+") as file:
            file_data = file.read()
        self.tree = RTFTree(file_data)
        print(self.tree)

#rtf = RTF("rtf/d")
rtf = RTF("rtf/file-sample_100kB.rtf")
