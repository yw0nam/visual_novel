import os
import io
import json


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def make_chat_template(instruction, input="", output="", system=""):
    inst_ls = [{"role": 'system', 'content': system}]
    inst_ls.append({"role": 'user', 'content': f"{instruction}\n\n{input}" if input != "" else instruction}),
    inst_ls.append({"role": 'assistant', 'content': output})
    return inst_ls

def bining_by_length(length, bin_1, bin_2, bin_3):
    if length <= bin_1:
        res = 1
    elif length > bin_1 and length <= bin_2:
        res = 2
    elif length > bin_2 and length <= bin_3:
        res = 3
    elif length > bin_3:
        res = 4
    return res

def get_length(chat_template):
    return sum(map(lambda x: len(x['content']) , chat_template))