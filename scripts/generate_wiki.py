#!/usr/bin/env python
"""
Wiki documentation generator for PACHA package.

This script extracts docstrings from all Python modules in the pacha package
and generates Markdown files suitable for GitHub Wiki.
"""

import ast
import os
import sys
from pathlib import Path


def extract_docstring(node):
    """
    Extract docstring from an AST node.

    Parameters
    ----------
    node : ast.AST
        AST node to extract docstring from.

    Returns
    -------
    str or None
        The docstring if present, None otherwise.
    """
    if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None


def get_function_signature(node):
    """
    Get the signature of a function or method.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        Function node.

    Returns
    -------
    str
        Function signature string.
    """
    args = []
    defaults_offset = len(node.args.args) - len(node.args.defaults)

    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        # Add default value if present
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(node.args.defaults):
            default = node.args.defaults[default_idx]
            if isinstance(default, ast.Constant):
                if isinstance(default.value, str):
                    arg_str += f"='{default.value}'"
                else:
                    arg_str += f"={default.value}"
            elif isinstance(default, ast.Name):
                arg_str += f"={default.id}"
            else:
                arg_str += "=..."
        args.append(arg_str)

    # Handle *args
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")

    # Handle **kwargs
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    return f"{node.name}({', '.join(args)})"


def parse_module(filepath):
    """
    Parse a Python module and extract documentation.

    Parameters
    ----------
    filepath : str or Path
        Path to the Python file.

    Returns
    -------
    dict
        Dictionary containing module documentation.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    module_doc = {
        'docstring': extract_docstring(tree),
        'classes': [],
        'functions': []
    }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_doc = {
                'name': node.name,
                'docstring': extract_docstring(node),
                'methods': []
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip private methods except __init__
                    if item.name.startswith('_') and item.name != '__init__':
                        continue
                    method_doc = {
                        'name': item.name,
                        'signature': get_function_signature(item),
                        'docstring': extract_docstring(item)
                    }
                    class_doc['methods'].append(method_doc)
            module_doc['classes'].append(class_doc)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private functions
            if node.name.startswith('_'):
                continue
            func_doc = {
                'name': node.name,
                'signature': get_function_signature(node),
                'docstring': extract_docstring(node)
            }
            module_doc['functions'].append(func_doc)

    return module_doc


def format_docstring(docstring, indent=0):
    """
    Format a docstring for Markdown output.

    Parameters
    ----------
    docstring : str or None
        The docstring to format.
    indent : int
        Number of spaces to indent.

    Returns
    -------
    str
        Formatted docstring.
    """
    if not docstring:
        return "*No documentation available.*"

    lines = docstring.strip().split('\n')
    # Remove common leading whitespace
    if len(lines) > 1:
        # Find minimum indentation (excluding empty lines)
        min_indent = None
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                if min_indent is None or indent < min_indent:
                    min_indent = indent
        if min_indent is not None:
            dedented_lines = []
            for line in lines[1:]:
                if len(line) > min_indent:
                    dedented_lines.append(line[min_indent:])
                else:
                    dedented_lines.append(line)
            lines = [lines[0]] + dedented_lines

    return '\n'.join(lines)


def generate_module_markdown(module_name, module_doc, subpackage=None):
    """
    Generate Markdown documentation for a module.

    Parameters
    ----------
    module_name : str
        Name of the module.
    module_doc : dict
        Module documentation dictionary.
    subpackage : str, optional
        Name of the subpackage.

    Returns
    -------
    str
        Markdown content.
    """
    lines = []

    # Title
    if subpackage:
        full_name = f"pacha.{subpackage}.{module_name}"
    else:
        full_name = f"pacha.{module_name}"

    lines.append(f"# {full_name}")
    lines.append("")

    # Module docstring
    if module_doc['docstring']:
        lines.append(format_docstring(module_doc['docstring']))
        lines.append("")

    # Classes
    if module_doc['classes']:
        lines.append("## Classes")
        lines.append("")

        for cls in module_doc['classes']:
            lines.append(f"### `{cls['name']}`")
            lines.append("")
            if cls['docstring']:
                lines.append(format_docstring(cls['docstring']))
                lines.append("")

            if cls['methods']:
                lines.append("#### Methods")
                lines.append("")
                for method in cls['methods']:
                    lines.append(f"##### `{method['signature']}`")
                    lines.append("")
                    if method['docstring']:
                        lines.append(format_docstring(method['docstring']))
                    lines.append("")

    # Functions
    if module_doc['functions']:
        lines.append("## Functions")
        lines.append("")

        for func in module_doc['functions']:
            lines.append(f"### `{func['signature']}`")
            lines.append("")
            if func['docstring']:
                lines.append(format_docstring(func['docstring']))
            lines.append("")

    return '\n'.join(lines)


def generate_home_page(structure):
    """
    Generate the Home page for the wiki.

    Parameters
    ----------
    structure : dict
        Dictionary mapping subpackages to their modules.

    Returns
    -------
    str
        Markdown content for the Home page.
    """
    lines = []
    lines.append("# PACHA Documentation")
    lines.append("")
    lines.append("**Precipitation Analysis & Correction for Hydrological Applications**")
    lines.append("")
    lines.append("A Python package for precipitation data fusion and analysis, combining data from")
    lines.append("satellites, weather radars, rain gauges, and commercial microwave links for")
    lines.append("improved precipitation estimation.")
    lines.append("")
    lines.append("## Package Structure")
    lines.append("")

    for subpackage, modules in sorted(structure.items()):
        if subpackage == '':
            lines.append("### Core Modules")
        else:
            lines.append(f"### pacha.{subpackage}")
        lines.append("")

        for module in sorted(modules):
            if subpackage:
                page_name = f"{subpackage}.{module}".replace('.', '-')
                display_name = f"pacha.{subpackage}.{module}"
            else:
                page_name = module
                display_name = f"pacha.{module}"
            lines.append(f"- [[{display_name}|{page_name}]]")
        lines.append("")

    lines.append("## Quick Links")
    lines.append("")
    lines.append("- [Installation Guide](https://github.com/rzambranap/PACHA#installation)")
    lines.append("- [Quick Start](https://github.com/rzambranap/PACHA#quick-start)")
    lines.append("- [Contributing](https://github.com/rzambranap/PACHA#contributing)")
    lines.append("")

    return '\n'.join(lines)


def generate_sidebar(structure):
    """
    Generate the sidebar for the wiki.

    Parameters
    ----------
    structure : dict
        Dictionary mapping subpackages to their modules.

    Returns
    -------
    str
        Markdown content for the sidebar.
    """
    lines = []
    lines.append("### Navigation")
    lines.append("")
    lines.append("[[Home]]")
    lines.append("")

    for subpackage, modules in sorted(structure.items()):
        if subpackage == '':
            lines.append("**Core**")
        else:
            lines.append(f"**{subpackage}**")

        for module in sorted(modules):
            if subpackage:
                page_name = f"{subpackage}.{module}".replace('.', '-')
            else:
                page_name = module
            lines.append(f"- [[{module}|{page_name}]]")
        lines.append("")

    return '\n'.join(lines)


def main():
    """Generate wiki documentation for the PACHA package."""
    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    pacha_dir = repo_root / 'pacha'
    wiki_dir = repo_root / 'wiki'

    # Create wiki directory
    wiki_dir.mkdir(exist_ok=True)

    # Subpackages to document
    subpackages = [
        'L1_processing',
        'L2_processing',
        'L3_processing',
        'analysis',
        'data_sources',
        'merging',
        'utils',
        'visualisation'
    ]

    structure = {'': []}  # Empty string key for core modules

    # Process core module
    core_file = pacha_dir / 'core.py'
    if core_file.exists():
        module_doc = parse_module(core_file)
        if module_doc:
            structure[''].append('core')
            content = generate_module_markdown('core', module_doc)
            with open(wiki_dir / 'core.md', 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Generated: core.md")

    # Process subpackages
    for subpackage in subpackages:
        subpackage_dir = pacha_dir / subpackage
        if not subpackage_dir.exists():
            continue

        structure[subpackage] = []

        for py_file in sorted(subpackage_dir.glob('*.py')):
            if py_file.name == '__init__.py':
                continue

            module_name = py_file.stem
            module_doc = parse_module(py_file)

            if module_doc:
                structure[subpackage].append(module_name)
                content = generate_module_markdown(module_name, module_doc, subpackage)
                # Use dashes instead of dots for wiki page names
                page_name = f"{subpackage}-{module_name}.md"
                with open(wiki_dir / page_name, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Generated: {page_name}")

    # Generate Home page
    home_content = generate_home_page(structure)
    with open(wiki_dir / 'Home.md', 'w', encoding='utf-8') as f:
        f.write(home_content)
    print("Generated: Home.md")

    # Generate Sidebar
    sidebar_content = generate_sidebar(structure)
    with open(wiki_dir / '_Sidebar.md', 'w', encoding='utf-8') as f:
        f.write(sidebar_content)
    print("Generated: _Sidebar.md")

    print(f"\nWiki documentation generated in: {wiki_dir}")


if __name__ == '__main__':
    main()
