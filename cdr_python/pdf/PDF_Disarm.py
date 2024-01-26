import os

def remove_unwanted_chars(byte_str: bytes)-> bytes:
    new_byte_str = b''
    for byte in byte_str:
        if byte in range(0x20, 0x7f):
            new_byte_str += bytes([byte])
    new_byte_str = new_byte_str.replace(b'%%EOF', b'')
    new_byte_str = new_byte_str.replace(b'%EOF', b'')
    new_byte_str = new_byte_str.replace(b'EOF', b'')
    return new_byte_str

class Obj:
    raw_data = b''
    def __init__(self,data):
        self.raw_data = data

    def __repr__(self) -> str:
        return str(self.raw_data)

class Body:
    file_name = ""
    xref_dict = {}
    start_xref = -1
    obj_dict = {}

    def __init__(self, xref_dict: dict,start_xref: int, file_name) -> None:
        #xref list shood be id to offse
        self.file_name = file_name
        self.xref_dict = xref_dict
        self.start_xref = start_xref
        offset_list = []
        for i , j in xref_dict.items():
            offset_list.append([j,i])
        offset_list.sort()
        offset_list.append([start_xref,-1])
        obj_dict = {}
        #here
        with open(file_name, 'rb') as f:
            for i in range(len(offset_list)-1):
                f.seek(offset_list[i][0],0)
                obj_raw = f.read(offset_list[i+1][0]-offset_list[i][0])
                obj_dict[offset_list[i][1]] = obj_raw
        self.obj_dict = obj_dict
        self.create_obj()

    def create_obj(self):
        for i in self.obj_dict.keys():
            t = Obj(self.obj_dict[i])
            self.obj_dict[i] = t

    def remove_by_id(self, id: int):
        self.obj_dict.pop(id,"ofek")

    def remove_opt_from_id(self, id: int, opt: bytes):
        if not self.obj_dict.get(id) or id == -1:
            return ''
        data = self.obj_dict.get(id).raw_data
        data: bytes
        offset = data.find(opt)
        if offset == -1:
            return

        end_offset = offset
        brackets = 0
        arrows = 0
        ls = b'/[]<>'
        for i in range(offset, len(data)):
            if data[i] == ls[0] and brackets == 0 and arrows == 0 and i != offset:
                end_offset = i
                break
            elif data[i] == ls[1]:
                brackets += 1
            elif data[i] == ls[2]:
                brackets -= 1
            elif arrows == 0 and data[i] == ls[4]:
                end_offset = i
                break
            elif data[i] == ls[3]:
                arrows += 1
            elif data[i] == ls[4]:
                arrows -= 1
        data = data[0:offset] + data[end_offset:]
        
        self.obj_dict.get(id).raw_data = data

    def get_type(self, id):
        if not self.obj_dict.get(id) or id == -1:
            return ''
        data = self.obj_dict.get(id).raw_data
        if data.find(b'/Type'):
            x = data.find(b'/Type')
            y = data[x+1:].find(b'/')
            if y == -1:
                return ''

            y = y+x+1
            tmp_d = data
            z = -1
            for i in range(y+1 , len(tmp_d)):
                found = False
                ls = b'/\n \r<>'
                for j in ls:
                    if tmp_d[i] == j:
                        z = i
                        return tmp_d[y:z]
            return tmp_d[y:z]

    def replace_content(self, id , new_content):
        if not self.obj_dict.get(id) or id == -1:
            return

        tmp = new_content.decode()
        tmp = f"{id} 0 obj\n{tmp}\n"
        new_content = tmp.encode()
        self.obj_dict[id] = Obj(new_content)

    def data_bin(self):
        body_bin =b''
        for i in self.obj_dict.values():
            body_bin += i.raw_data
        body_bin += b'\n'
        return body_bin

class Xref:
    plain_obj_ref_arr = [] #of binary strings
    obj_dict_ref = {}
    obj_list_ref = []
    start_xref = 0
    def __init__(self, obj_ref, start_xref) -> None:
        self.plain_obj_ref_arr = obj_ref
        self.start_xref = start_xref

    def create_xref_dict(self):
        temp_xref = self.plain_obj_ref_arr
        for i in range(len(temp_xref)):
            temp_xref[i] = temp_xref[i].split(b' ')
            rem_list =[]
            for j in range(len(temp_xref[i])):
                if len(temp_xref[i][j]) == 0:
                    rem_list.append(j)
            
            tmp_list = []
            for j in range(len(temp_xref[i])):
                if j not in rem_list:
                    tmp_list.append(temp_xref[i][j])       
            temp_xref[i] = tmp_list        
            if len(temp_xref[i]) == 2 or len(temp_xref[i]) == 3:
                for cell in range(len(temp_xref[i])):
                    if temp_xref[i][cell] == b'f' or temp_xref[i][cell] == b'n':
                        temp_xref[i][cell] = temp_xref[i][cell]
                    elif temp_xref[i][cell].decode().isdigit():
                        temp_xref[i][cell] = int(temp_xref[i][cell])
                    else:
                        raise Exception(f"xref entery is not right {temp_xref[i]}")
            else:
                raise Exception(f"not an xref entery {temp_xref[i]}")
        
        start_index = 0
        obj_runner = 0
        xref_list = []
        for cell in temp_xref:
            if len(cell) == 2:
                start_index = cell[0]
                obj_runner = 0
            else:
                xref_list.append([start_index + obj_runner,cell])
                obj_runner += 1 
        if self.start_xref <  xref_list[-1][1][0]:
            #probbly a type of pdf that are knon - in xref they add 10 before number so we try to fix it
            for i in range(1,len(xref_list)):
                xref_list[i][1][0] = int(str(xref_list[i][1][0])[2:])     
        xref_dict = {}
        for cell in xref_list:
            xref_dict[cell[0]] = cell[1][0]
        
        
        self.obj_dict_ref = xref_dict
        self.obj_list_ref = xref_list

    def object_by_offset(self, offset: int) -> int:
        if offset < 0:
            print("WARNING - OFFSET IS NEGETIVE")
            return -1
        for i in range(len(self.obj_list_ref)):
            if self.obj_list_ref[i][1][0] > offset:
                return self.obj_list_ref[i-1][0]
        return self.obj_list_ref[len(self.obj_list_ref)-1][0]

    def __repr__(self) -> str:
        string_xref = 'xref\n'
        for i in  range(len(self.obj_list_ref)):
            string_xref += str(self.obj_list_ref[i][0])
            string_xref += " 1\n"
            string_offset = "0"*(10-len(str(self.obj_list_ref[i][1][0])))
            string_offset += str(self.obj_list_ref[i][1][0])
            string_xref += string_offset
            string_xref += " "
            string_offset = "0"*(5-len(str(self.obj_list_ref[i][1][1])))
            string_offset += str(self.obj_list_ref[i][1][1])
            string_xref += string_offset
            string_xref += " "
            string_xref += str(self.obj_list_ref[i][1][2].decode())
            string_xref += "\n"
        return string_xref
class Trailer:
    plain_data = b''
    size = b''
    root = b''
    info = b''
    #id = b''

    def __init__(self, trailer_data: bytes) -> None:
        self.plain_data = trailer_data
        trailer_data = trailer_data.replace(b'<<', b'')
        trailer_data = trailer_data.replace(b'>>', b'')
        trailer_data = trailer_data.split(b'/') #split all the trailer options
        trailer_data = trailer_data[1:] #remove empty 
        for i in trailer_data:
            i: bytes
            if b'Size' in i:
                self.size = i
            if b'Root' in i:
                self.root = i
            if b'Info' in i:
                self.info = i

    def get_info(self)->int:
        temp_str = self.info.replace(b'Info', b'')
        temp_str = temp_str.decode("utf-8").lstrip()
        if not temp_str.split(" ")[0].isdigit():
            raise Exception('info id is not an integer, cant return it')
        return temp_str.split(" ")[0]

    def get_root(self)->int:
        temp_str = self.root.replace(b'Root', b'')
        temp_str = temp_str.decode("utf-8").lstrip()
        if not temp_str.split(" ")[0].isdigit():
            raise Exception('root id is not an integer, cant return it')
        return temp_str.split(" ")[0]

    def get_size(self)->int:
        temp_str = self.size.replace(b'Size', b'')
        if not temp_str.decode("utf-8").lstrip().isdigit():
            raise Exception('size is not an integer, cant return it')
        return int(temp_str.decode("utf-8"))
    
    def set_info(self, id: int):
        new_info = b'Info '+ str(id).encode() + b' 0 R'
        self.info = new_info

    def set_root(self, id):
        new_root = b'Root '+ str(id).encode() + b' 0 R'
        self.root = new_root

    def set_size(self, id):
        new_size = b'Size '+ str(id).encode()
        self.size = new_size

    def get_trailer_b(self):
        return b'trailer\r\n<<\r\n/' + self.size + b'\r\n/' + self.root + b'\r\n/' + self.info + b'\r\n>>\r\n'

    def __repr__(self) -> str:
        return (b'trailer\r\n<<\r\n/' + self.size + b'\r\n/' + self.root + b'\r\n/' + self.info + b'\r\n>>\r\n').decode()

class PdfFile:
    file_name = '' #a string
    trailer_str = b'' #a binery string
    xref_str = b'' #a binary string

    trailer_obj = None
    xref_obj = None
    body_obj = None

    start_xref = -1
    start_trailer = -1
    def __init__(self, name: str) -> None:
        self.file_name = name 
        if self.file_name == '':
            raise Exception('File name is empty')
        if not self.file_name.endswith('.pdf'):
            raise Exception('File name is not in pdf format')
        self.create_trailer()
        self.create_xref()
        self.create_body()
        self.refrash_file()
    def create_body(self):
        self.body_obj = Body(self.xref_obj.obj_dict_ref, self.start_xref, self.file_name)

    def create_xref(self):
        self.find_startxref()
        if type(self.start_xref) != int:
            raise Exception("start xref is not a number")
        if self.start_xref == -1:
            raise Exception("xref is -1")

        xref_objects_refrence = []
        with open(self.file_name, 'rb') as f:
            f.seek(self.start_xref, 0)
            xref = f.read(self.start_trailer - self.start_xref)
            xref = xref.replace(b'\r', b'\n')
            xref = xref.replace(b'\n\n', b'\n')
            xref_temp = xref.split(b'\n')
            for i in range(len(xref_temp)):
                xref_temp[i] = remove_unwanted_chars(xref_temp[i])
            if len(xref_temp) < 3:
                raise Exception("not a legel xref structure")
            if not xref_temp[0] == b'xref':
                seeker = self.start_trailer
                f.seek(seeker,os.SEEK_SET)
                fixed = False
                while seeker > 0:
                    x = f.read(4)
                    if x == b'xref':
                        fixed = True
                        self.start_xref = seeker
                        f.seek(self.start_xref, 0)
                        xref = f.read(self.start_trailer - self.start_xref)
                        xref = xref.replace(b'\r', b'\n')
                        xref = xref.replace(b'\n\n', b'\n')
                        xref_temp = xref.split(b'\n')
                        for i in range(len(xref_temp)):
                            xref_temp[i] = remove_unwanted_chars(xref_temp[i])
                    else:
                        seeker -= 1
                        f.seek(seeker,os.SEEK_SET)
                if not fixed:
                    raise Exception("not a legel pdf structure - non xref table start")
            if not len(xref_temp[1].decode().lstrip().rstrip().encode().split(b' ')) == 2:
                raise Exception("not a legel xref structure")
            xref_objects_refrence = xref_temp[1:]
        xref_objects_refrence_temp = []
        for i in range(len(xref_objects_refrence)):
            if len(xref_objects_refrence[i]) > 2:
                xref_objects_refrence_temp.append(xref_objects_refrence[i])
        xref_objects_refrence = xref_objects_refrence_temp
        self.xref_obj = Xref(xref_objects_refrence,self.start_xref)
        self.xref_obj.create_xref_dict()

    def create_trailer(self):
        try:
            self.find_trailer()
        except:
            trailer_str_tmp = 'trailer<< /Size 1 /Root 1 0 R /Info 1 0 R >>'
            self.trailer_str = trailer_obj
        trailer_obj = Trailer(self.trailer_str)
        self.trailer_obj = trailer_obj

    def find_trailer(self):
        buff_size = 7
        b_string_to_find = b'trailer'
        start_offset = self.find_key_word(buff_size, b_string_to_find)
        start_offset += buff_size
        buff_size = 2
        b_string_to_find = b'>>'
        end_offset = self.find_key_word(buff_size, b_string_to_find)
        end_offset += 2
        with open(self.file_name, "rb") as f:
            f.seek(start_offset,0)
            trailer = f.read(end_offset - start_offset)
            trailer = remove_unwanted_chars(trailer)
        self.start_trailer = start_offset - 7
        self.trailer_str = trailer

    def find_startxref(self):
        buff_size = 9
        b_string_to_find = b'startxref'
        offset = self.find_key_word(buff_size, b_string_to_find)
        content = b''
        with open(self.file_name, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(offset+9,0)
            content = f.read(file_size - offset)
        start_xref = remove_unwanted_chars(content).decode("utf-8")
        if start_xref.isdigit():
            start_xref = int(start_xref)
        else:
            raise ValueError("Xerf offset is not a number")
        self.start_xref = start_xref

    def find_key_word(self, buff_size: int, b_string_to_find: bytes):
        with open(self.file_name, "rb") as f:
            f.seek(0, os.SEEK_END)
            offset = f.tell()
            f.seek(offset,0)
            while offset > 0:
                b_content = f.read(buff_size)
                if b_content == b_string_to_find:
                    return offset
                offset -= 1
                f.seek(offset,0)
            raise Exception(f'{b_string_to_find} NOT found')

    def get_id_by_offset(self, offset: int)-> int:
        return self.xref_obj.object_by_offset(offset)

    def remove_obj_by_id(self, id: int, cdr_rule_dict):
        if self.trailer_obj.get_root() != b'':
            if id == int(self.trailer_obj.get_root()): 
                get_rule = cdr_rule_dict["root"]
                if get_rule[0] == 'r':
                    for i in get_rule[1]:
                        self.body_obj.remove_opt_from_id(id, i)
                return
        get_t = self.body_obj.get_type(id)
        if type(get_t) == bytes:
            try:
                get_t = get_t.decode()
            except Exception as e:
                get_t = ""
        get_rule = cdr_rule_dict.get(get_t)
        if not get_rule:
            get_rule = cdr_rule_dict.get("default")
        if get_rule[0] == 'ra':
            self.body_obj.remove_by_id(id)
        elif get_rule[0] == 'c':
            self.body_obj.replace_content(id, get_rule[1][0])
        elif get_rule[0] == 'r':
            for i in get_rule[1]:
                self.body_obj.remove_opt_from_id(id, i)
        print(get_rule[0])
        self.refrash_file()

    def refrash_file(self):
        #need some more work but fine for testind - need to add update to start xref and trailer and raw data
        body = self.body_obj.obj_dict
        temp_xref_list = []
        offset = 0
        self.xref_obj.obj_dict_ref = self.body_obj.xref_dict
        for i , j in body.items():
            if i == 0:
                temp_xref_list.append([i,[offset, 65535, b'f']]) 
            else:
                temp_xref_list.append([i,[offset, 0, b'n']]) 
            offset +=len(j.raw_data)
        self.xref_obj.obj_list_ref = temp_xref_list
        self.trailer_obj.set_size(len(temp_xref_list))
        self.start_xref = len(self.body_obj.data_bin())

    def dump_file(self,path, new_name = " "):
        self.refrash_file()
        raw_data = self.body_obj.data_bin()
        raw_data += str(self.xref_obj).encode()
        raw_data += str(self.trailer_obj).encode()
        raw_data += b'startxref\n'
        raw_data += str(self.start_xref).encode()
        raw_data += b'\n%%EOF'
        new_file_name = path + "/" + new_name
        with open(new_file_name, 'wb+') as nf:
            nf.write(raw_data)


"""
def main():
    pdf = PdfFile("bad_files/read_length_must_be_non-negative_or_-1/0b73da0d940b8c0657563cb4e0c6a3304f649791fc96d08d4c0f451dcc49f20a.pdf")
    pdf.dump_file("./TEMP_pdf", "name1.pdf")
main()"""