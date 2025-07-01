#!/usr/bin/env python3
"""
File utility functions shared across multiple modules.
"""

import os
import platform


def ensure_directory_exists(directory):
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory

    Returns:
        str: Absolute path to the directory
    """
    if directory:
        directory = os.path.abspath(directory)
        os.makedirs(directory, exist_ok=True)
    return directory


def get_file_extension(file_path):
    """Get the extension of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: File extension (lowercase, without the dot)
    """
    return os.path.splitext(file_path)[1].lower()[1:]


def find_executable(exec_name, possible_paths=None):
    """Find an executable in common locations or PATH.

    Args:
        exec_name (str): Name of the executable
        possible_paths (list): List of possible paths to check

    Returns:
        str: Path to executable or original name if not found in specific paths
    """
    # Default to the executable name, which will use PATH
    executable = exec_name

    # On macOS, check specific paths if provided
    if platform.system() == "Darwin" and possible_paths:
        for path in possible_paths:
            if os.path.exists(path):
                executable = path
                break

    return executable
