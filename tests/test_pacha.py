"""
Unit tests for PACHA package.

This module contains basic unit tests to verify the package installation
and core functionality.
"""

import unittest


class TestStringMethods(unittest.TestCase):
    """Basic string method tests (placeholder)."""

    def test_upper(self):
        """Test uppercase conversion."""
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        """Test isupper method."""
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        """Test string split method."""
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


class TestPachaImport(unittest.TestCase):
    """Test PACHA package imports."""

    def test_import_pacha(self):
        """Test that pacha package can be imported."""
        import pacha
        self.assertIsNotNone(pacha.__version__)

    def test_import_utils(self):
        """Test that utils subpackage can be imported."""
        from pacha import utils
        self.assertIsNotNone(utils)


if __name__ == '__main__':
    unittest.main()
