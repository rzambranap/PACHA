"""
Unit tests for wiki generation script.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from generate_wiki import (
    extract_docstring,
    get_function_signature,
    parse_module,
    format_docstring,
    generate_module_markdown
)


class TestExtractDocstring(unittest.TestCase):
    """Test docstring extraction functions."""

    def test_extract_docstring_from_source(self):
        """Test extracting docstring from a module."""
        import ast
        source = '''
"""Module docstring."""

def foo():
    """Function docstring."""
    pass
'''
        tree = ast.parse(source)
        docstring = extract_docstring(tree)
        self.assertEqual(docstring, "Module docstring.")

    def test_extract_docstring_no_docstring(self):
        """Test handling of module without docstring."""
        import ast
        source = '''
def foo():
    pass
'''
        tree = ast.parse(source)
        docstring = extract_docstring(tree)
        self.assertIsNone(docstring)


class TestGetFunctionSignature(unittest.TestCase):
    """Test function signature extraction."""

    def test_simple_function(self):
        """Test signature of simple function."""
        import ast
        source = '''
def foo(a, b):
    pass
'''
        tree = ast.parse(source)
        func_node = tree.body[0]
        sig = get_function_signature(func_node)
        self.assertEqual(sig, "foo(a, b)")

    def test_function_with_defaults(self):
        """Test signature with default values."""
        import ast
        source = '''
def foo(a, b=10, c='hello'):
    pass
'''
        tree = ast.parse(source)
        func_node = tree.body[0]
        sig = get_function_signature(func_node)
        self.assertEqual(sig, "foo(a, b=10, c='hello')")

    def test_function_with_args_kwargs(self):
        """Test signature with *args and **kwargs."""
        import ast
        source = '''
def foo(a, *args, **kwargs):
    pass
'''
        tree = ast.parse(source)
        func_node = tree.body[0]
        sig = get_function_signature(func_node)
        self.assertEqual(sig, "foo(a, *args, **kwargs)")


class TestParseModule(unittest.TestCase):
    """Test module parsing."""

    def test_parse_module_with_functions(self):
        """Test parsing a module with functions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''"""Test module."""

def public_func():
    """Public function docstring."""
    pass

def _private_func():
    """Private function docstring."""
    pass
''')
            temp_path = f.name

        try:
            result = parse_module(temp_path)
            self.assertEqual(result['docstring'], "Test module.")
            self.assertEqual(len(result['functions']), 1)  # Only public function
            self.assertEqual(result['functions'][0]['name'], 'public_func')
        finally:
            os.unlink(temp_path)

    def test_parse_module_with_classes(self):
        """Test parsing a module with classes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''"""Test module."""

class MyClass:
    """Class docstring."""
    
    def __init__(self):
        """Init docstring."""
        pass
    
    def public_method(self):
        """Public method."""
        pass
    
    def _private_method(self):
        """Private method."""
        pass
''')
            temp_path = f.name

        try:
            result = parse_module(temp_path)
            self.assertEqual(len(result['classes']), 1)
            self.assertEqual(result['classes'][0]['name'], 'MyClass')
            # Should have __init__ and public_method, not _private_method
            method_names = [m['name'] for m in result['classes'][0]['methods']]
            self.assertIn('__init__', method_names)
            self.assertIn('public_method', method_names)
            self.assertNotIn('_private_method', method_names)
        finally:
            os.unlink(temp_path)


class TestFormatDocstring(unittest.TestCase):
    """Test docstring formatting."""

    def test_format_none(self):
        """Test formatting None docstring."""
        result = format_docstring(None)
        self.assertEqual(result, "*No documentation available.*")

    def test_format_simple_docstring(self):
        """Test formatting simple docstring."""
        result = format_docstring("Simple docstring.")
        self.assertEqual(result, "Simple docstring.")

    def test_format_multiline_docstring(self):
        """Test formatting multiline docstring."""
        docstring = """First line.

        Second line.
        Third line."""
        result = format_docstring(docstring)
        self.assertIn("First line.", result)
        self.assertIn("Second line.", result)


class TestGenerateModuleMarkdown(unittest.TestCase):
    """Test markdown generation."""

    def test_generate_markdown(self):
        """Test generating module markdown."""
        module_doc = {
            'docstring': 'Test module docstring.',
            'classes': [],
            'functions': [
                {
                    'name': 'test_func',
                    'signature': 'test_func(a, b)',
                    'docstring': 'Test function docstring.'
                }
            ]
        }
        result = generate_module_markdown('test_module', module_doc, 'utils')
        self.assertIn('# pacha.utils.test_module', result)
        self.assertIn('Test module docstring.', result)
        self.assertIn('## Functions', result)
        self.assertIn('### `test_func(a, b)`', result)
        self.assertIn('Test function docstring.', result)


if __name__ == '__main__':
    unittest.main()
