import os
import re

def elide_string(s, n=80):
    return s[:min(len(s),n-3)]+'...'

def get_sanitized_filename(full_path: str, max_length: int = 200) -> str:
    """
    Sanitizes a filename by removing/replacing invalid characters, truncating it
    to a specified maximum length, and adding an incrementing number before
    the extension to prevent overwriting existing files. It checks for existence
    within the parent directory of the provided full_path.

    Args:
        full_path (str): The full desired path, including the original filename.
                         The existence check will be performed in its parent directory.
        max_length (int): The maximum allowed length for the base filename
                          (before extension and incrementing number).
                          Default is 200 characters, leaving room for
                          extension and a potential increment.

    Returns:
        str: A unique, sanitized, and truncated file path
    """
    # 1. Separate directory, filename, and extension
    target_directory = os.path.dirname(full_path)
    original_filename = os.path.basename(full_path)

    base_name, extension = os.path.splitext(original_filename)

    # Remove leading/trailing whitespace from base_name and extension
    base_name = base_name.strip()
    extension = extension.strip()

    # 2. Sanitize special characters
    # Replace invalid characters (common across Windows/Unix) with an underscore
    # Characters like / \ : * ? " < > | are generally problematic.
    # Also, control characters and non-printable ASCII.
    sanitized_base_name = re.sub(r'[\\/:*?"<>|\x00-\x1F]', '_', base_name)

    # Remove any dots that might appear at the start or end of the base name
    # after sanitization, which could interfere with extension parsing.
    sanitized_base_name = sanitized_base_name.strip('.')

    # Replace multiple underscores with a single one
    sanitized_base_name = re.sub(r'_{2,}', '_', sanitized_base_name)

    # 3. Truncate to a specific length, reserving space for potential increment
    # We will reserve space for a counter up to 9999 (i.e., "_9999" which is 5 characters).
    MAX_COUNTER_SUFFIX_LEN = 5 # For "_9999"

    # Ensure max_length is sufficient to hold at least a minimal name,
    # the maximum counter suffix, and the extension.
    # e.g., "a_9999.txt" needs at least 11 chars (1 for min base char + 5 for suffix + 4 for .txt + 1 for dot)
    MIN_REQUIRED_LENGTH = 1 + MAX_COUNTER_SUFFIX_LEN + len(extension) + 1
    if max_length < MIN_REQUIRED_LENGTH:
        max_length = MIN_REQUIRED_LENGTH
        print(f"Warning: max_length was too small. Adjusted to {max_length}.")

    # Calculate effective max length for the base name, reserving space for the largest possible counter suffix
    effective_max_base_length = max_length - len(extension) - MAX_COUNTER_SUFFIX_LEN - 1 # -1 for the dot

    # Truncate the base name once, upfront
    if len(sanitized_base_name) > effective_max_base_length:
        sanitized_base_name = sanitized_base_name[:effective_max_base_length]

    # 4. Add an incrementing number before the extension to prevent overwriting
    counter = 0
    # Initial proposed filename without counter (might be unique already)
    final_filename_candidate = f"{sanitized_base_name}{extension}"
    full_check_path = os.path.join(target_directory, final_filename_candidate)

    # Check if the initial candidate exists. If so, start incrementing.
    if os.path.exists(full_check_path):
        while True:
            counter += 1
            counter_suffix = f"_{counter}"

            # Safety break to prevent infinite loops and excessively long counters
            if counter > 9999: # Cap the counter at 9999 as per MAX_COUNTER_SUFFIX_LEN assumption
                print(f"Warning: Could not find a unique filename after {counter} attempts. Max counter reached.")
                # As a last resort, return the last attempted name, though it might conflict
                break # Exit the while loop

            # The sanitized_base_name is already truncated to accommodate the max counter suffix.
            # So, we just append the current counter suffix.
            final_filename_candidate = f"{sanitized_base_name}{counter_suffix}{extension}"
            full_check_path = os.path.join(target_directory, final_filename_candidate)

            if not os.path.exists(full_check_path):
                break # Found a unique filename

    return os.path.join(target_directory,final_filename_candidate)