
import os

import toml


def write_dict_as_toml(dictionary: dict, directory: str, filename: str):
    """General purpose dictionary saving method.

    :param dictionary:
        Dictionary to be written as toml.
    :param directory:
        Path to the directory where the toml should be written.
    :param filename:
        Name of the file to be written. Postfix '.toml' will be added if necessary.
    """

    if not os.path.exists(os.path.abspath(directory)):
        raise RuntimeError(f"Cannot write given dictionary to path '{os.path.abspath(directory)}' "
                           f"because it does not exist.")

    if not filename.endswith('.toml'):
        filename = filename + '.toml'

    p = os.path.join(directory, filename)
    with open(p, 'w+') as file:
        toml.dump(dictionary, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_toml_as_dict(directory: str, filename: str):
    """General purpose toml reading method.

    :param directory:
        Path to the directory where the toml file is.
    :param filename:
        Name of the file to be read. Postfix '.toml' will be added if necessary.
    :return dictionary:
        Returns read toml file as a dictionary.
    """

    if not filename.endswith('.toml'):
        filename = filename + '.toml'

    p = os.path.join(directory, filename)

    if not os.path.exists(os.path.abspath(p)):
        raise FileNotFoundError(f"Cannot read from file '{os.path.abspath(p)}' "
                           f"because it does not exist.")

    with open(p, 'r') as file:
        result = toml.load(file)
    return result
