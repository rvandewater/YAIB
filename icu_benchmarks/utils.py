import argparse
import json


def parse_dict(arg):
    """
    Parses a string into a dictionary. Handles both:
    - Unquoted format: 'key1:value1,key2:value2'
    - JSON-like quoted format: '"key1":"value1","key2":"value2"'
    """
    try:
        # Check if the input is in JSON-like format
        if ":" in arg and '"' in arg:
            # Wrap in curly braces to make it valid JSON
            json_string = f"{{{arg}}}"
            return json.loads(json_string)
        else:
            # Handle unquoted format
            pairs = arg.split(',')
            return {key.strip(): value.strip() for key, value in (pair.split(':', 1) for pair in pairs)}
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")
