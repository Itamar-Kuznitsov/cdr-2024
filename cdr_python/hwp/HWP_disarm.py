import hwp5
from hwp5.filestructure import Hwp5File
import yara



class HwpFile(Hwp5File):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.hwp_file = Hwp5File(file_path)
        a =1

    def get_id_by_offset(self, offset: int) -> int:
        for obj in self.hwp_file.bodytext.sections:
            if obj.offset == offset:
                return obj.id
        return -1
    
    def remove_obj_by_id(self, id: int, cdr_rule_dict):
        # Compile the YARA rules
        rules = yara.compile(source=cdr_rule_dict)

        # Iterate over the sections in the HWP file
        for section in self.hwp_file.bodytext.sections:
            # Check if the object ID matches the provided ID
            if section.id == id:
                # Apply the YARA rules to the object's data
                matches = rules.match(data=section.data)

                # If there are matches, remove the object
                if matches:
                    self.hwp_file.bodytext.sections.remove(section)
                    print(f"Removed object with ID {id}")

# Open the HWP file and create an HwpFile object
hwp_file = HwpFile("/Users/itamar/Documents/CS/final_project_2024/한글문서파일형식3.0_HWPML_revision1.2.hwp")

# Now, you can use the 'hwp_file' object to access various properties and data
print(f'HWP file version: {hwp_file.header.version}')

# TODO(IK): make a new class which inherits from Hwp5File and add more functionality like in the pdf_disarm_v2.py
