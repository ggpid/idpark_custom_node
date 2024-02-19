import shutil
import folder_paths
import os
import sys
import traceback

module_path = os.path.join(os.path.dirname(__file__))
sys.path.append(module_path)

from codewave_nodes import SAMGenerator

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SAMGenerator": SAMGenerator,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMGenerator": "SAMGenerator Node",
}
