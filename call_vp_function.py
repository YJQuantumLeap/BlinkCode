import os
import sys
from utils import chdir, time_limit
IMPORT_viper = True

if IMPORT_viper:
    sys.path.insert(0, os.getcwd() + '/viper')
    with chdir('./viper'):
        from main_simple_lib import *
def run_vp_code(code, image_path, timeout):#abs path
    if code == "":
        return "Error:the code is empty!"
    if "def execute_command" not in code:
        return "Error:the definition for the function \"def execute_command\" is not found."
    with chdir('./viper'):
        with time_limit(timeout):
            # with time_limit(300):
            im = load_image(image_path)
            try:
                # print(f"now code\n{code}")
                exec(code, globals())
                result = str(execute_command(im))
                # print(f'\nexec resuult: {result}')
            except Exception as e:
                result = "Error:" + str(e)
            return str(result)

if __name__ == "__main__":
    code = """
def execute_command(image) -> str:
    image_patch = ImagePatch(image)
    muffin_patches = image_patch.find("muffin")
    n_muffins = len(muffin_patches)
    kid_patches = image_patch.find("kid")
    n_kids = len(kid_patches)
    result = str(n_muffins // n_kids)
    return result
# def execute_command(im):
#     image_patch = ImagePatch(im)
#     return str(image_patch.simple_query("What's in the picture")) + image_patch.OCR()
"""
    image_path = os.getcwd() + "/viper/image.png"
    timeout = 300
    print(run_vp_code(code, image_path, timeout))