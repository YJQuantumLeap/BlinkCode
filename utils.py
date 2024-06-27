import base64
import asyncio
import random
from typing import Optional
import contextlib
import faulthandler
import io
import signal
import tempfile
import shutil
import resource
import builtins
import subprocess
import sys
import json
import os
import imageio
from PIL import Image
import re
from typing import Iterable, Dict
import gzip
import imgkit


def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_jsonl(file_path) -> list:
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]


def get_code(s: str, start='```python\n', end='```', is_contain_start_end=False):
    # Find the first substring in s that starts with 'start' and ends with 'end'.
    start_index = s.find(start)
    if start_index == -1:
        return ""
    end_index = s.find(end, start_index + len(start))
    if end_index == -1:
        return ""
    idx = len(start)
    if is_contain_start_end == False:
        return s[start_index + idx:end_index].strip()
    else:
        return s[start_index:end_index + len(end)].strip()


def get_python_code(s: str):
    code = get_code(s)
    if code == "":
        code = get_code(s, '```py', '```')
    if code == "":
        code = get_code(s, '```', '```')
    if code == "" and (len(s) >= 3 and s[0:3] == "def"):
        code = s
    if code == "" and ((len(s) >= 6 and s[0:6] == "import") or (len(s) >= 4 and s[0:4] == "from")):
        code = s
    if code == "" and len(s) > 0 and s[0] == "#":
        code = s
    return code


def get_html_code(s: str):
    html = get_code(s, start="<!DOCTYPE html>",
                    end="</html>", is_contain_start_end=True)
    if html == "":
        html = get_code(s, start="<html>", end="</html>",
                        is_contain_start_end=True)
    return html


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def process_figure_code(code, image_path):
    def delete_a_line_contain_ss(content: str, ss: str):
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if ss not in line:
                new_lines.append(line)
        new_content = '\n'.join(new_lines)
        return new_content
    content = delete_a_line_contain_ss(code, '.show()')
    content = "import matplotlib.pyplot as plt\nimport matplotlib\nmatplotlib.use(\'Agg\')\n" + \
        content + f"\nplt.savefig(\'{image_path}\')"
    return content


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


class ReliabilityGuard:
    def __init__(self, maximum_memory_bytes: Optional[int] = None):
        self.maximum_memory_bytes = maximum_memory_bytes
        self.original_settings = {}

    def __enter__(self):
        if self.maximum_memory_bytes is not None:
            self.set_memory_limits()
        self.disable_functions()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_functions()

    def set_memory_limits(self):
        resource.setrlimit(
            resource.RLIMIT_AS, (self.maximum_memory_bytes, self.maximum_memory_bytes))
        resource.setrlimit(
            resource.RLIMIT_DATA, (self.maximum_memory_bytes, self.maximum_memory_bytes))
        if not sys.platform.startswith('darwin'):
            resource.setrlimit(
                resource.RLIMIT_STACK, (self.maximum_memory_bytes, self.maximum_memory_bytes))

    def disable_functions(self):
        faulthandler.disable()
        self.original_settings['exit'] = builtins.exit
        self.original_settings['quit'] = builtins.quit
        builtins.exit = None
        builtins.quit = None

        self.original_settings['os_environ'] = os.environ.copy()
        os.environ['OMP_NUM_THREADS'] = '1'

        os_functions = [
            'kill', 'system', 'remove', 'removedirs', 'rmdir', 'fchdir', 'setuid', 'fork', 'forkpty',
            'killpg', 'rename', 'renames', 'truncate', 'replace', 'unlink', 'fchmod', 'fchown', 'chmod', 'chown',
            'chroot', 'fchdir', 'lchflags', 'lchmod', 'lchown'
        ]
        for func_name in os_functions:
            if hasattr(os, func_name):
                self.original_settings[func_name] = getattr(os, func_name)
                setattr(os, func_name, None)

        shutil_functions = ['rmtree', 'move', 'chown']
        for func_name in shutil_functions:
            if hasattr(shutil, func_name):
                self.original_settings[func_name] = getattr(shutil, func_name)
                setattr(shutil, func_name, None)

        if hasattr(subprocess, 'Popen'):
            self.original_settings['Popen'] = subprocess.Popen
            subprocess.Popen = None

        sys_modules = ['ipdb', 'joblib', 'resource', 'psutil', 'tkinter']
        for module_name in sys_modules:
            if module_name in sys.modules:
                self.original_settings[module_name] = sys.modules.get(
                    module_name)
                sys.modules[module_name] = None

    def restore_functions(self):
        builtins.exit = self.original_settings.get('exit')
        builtins.quit = self.original_settings.get('quit')

        for key, value in self.original_settings.get('os_environ', {}).items():
            os.environ[key] = value

        for func_name, func in self.original_settings.items():
            if func_name in ['exit', 'quit', 'os_environ']:
                continue
            if hasattr(os, func_name):
                setattr(os, func_name, func)
            elif hasattr(shutil, func_name):
                setattr(shutil, func_name, func)
            elif func_name == 'Popen':
                subprocess.Popen = func
            elif func_name in sys.modules:
                sys.modules[func_name] = func


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


def unsafe_execute_matplotlib_code(code):
    # It is best to call this function in a temporary directory
    try:
        exec_globals = {}
        with time_limit(3):
            exec(code, exec_globals)
        return "True"
    except TimeoutException:
        return "time-out"
    except BaseException as e:
        return f"{e}"


def unsafe_execute_code(code):
    with create_tempdir():
        with ReliabilityGuard():
            try:
                exec_globals = {}
                with time_limit(3):
                    exec(code, exec_globals)
                return "True"
            except TimeoutException:
                return "time-out"
            except AssertionError:
                return "Your code has no syntax errors, but it doesn't pass all the test cases"
            except BaseException as e:
                return f"{e}"


def cal_evaluation_score(L: list):
    if len(L) != 3:
        return 'error, len(L) != 3'
    mp = [0, 0, 0, 0, 0]
    for i in L:
        mp[i] += 1
    if mp[3] == 3 or (mp[3] == 2 and mp[2] == 1):
        return 4
    if (mp[3] + mp[2] >= 3):
        return 3
    if (mp[3] + mp[2] >= 2) and mp[0] == 0:
        return 2
    if mp[3] + mp[2] + mp[1] >= 2:
        return 1
    return 0


def convert_png_to_jpeg(source_path, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    img = Image.open(source_path)
    if img.mode == 'RGBA' or img.mode == 'P':
        img = img.convert('RGB')
    img.save(target_path, 'JPEG')
    return target_path


def run_pdflatex(input_file, output_dir="output"):
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-file-line-error",
        f"-output-directory={output_dir}",
        input_file
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    error_pattern = re.compile(r"^.*:(\d+): (.*)$")

    error_set = set()

    for line in stdout.splitlines():
        match = error_pattern.match(line.strip())
        if match:
            line_number = match.group(1)
            error_message = match.group(2)
            error_set.add((line_number, error_message))

    error_mess = "error message:\n"
    if error_set:
        for line_number, error_message in sorted(error_set):
            error_mess = error_mess + f"Line {line_number}: {error_message}\n"
        return error_mess
    return "True"


def check_and_process_images(file_path):
    file_size = os.path.getsize(file_path)
    with Image.open(file_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(file_path)
    while file_size > 5 * 1024 * 1024:
        print(
            f"Compressing and converting: {file_path} (size: {file_size / 1024 / 1024:.2f} MB)")
        with Image.open(file_path) as img:
            img.save(file_path, quality=80)
        file_size = os.path.getsize(file_path)

def check_image_path(image_path1, image_path2):
    if image_path1 != None and image_path2 != None:
        return True
    if image_path1 != None and image_path2 == None:
        return True
    if image_path1 == None and image_path2 == None:
        return True
    return False


def convert_html_to_image(html_path, image_path):
    cfg = imgkit.config()
    options = {
        'quality': 100,
        'zoom': 1.0,
        'encoding': 'UTF-8',
        'no-images': ''
    }
    imgkit.from_file(html_path, image_path, config=cfg, options=options)


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


if __name__ == '__main__':
    pass
    # convert_html_to_image()
