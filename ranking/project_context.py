"""Workspace Project Context Parser and Embedding Generator.

Parses dependency files (pyproject.toml, package.json, requirements.txt) and workspace file extensions
to generate a project context embedding using the sentence-transformers encoder.
"""

import os
import json
import logging
from typing import List, Set, Dict, Optional
from retrieval.embedder import get_embedding_model

logger = logging.getLogger(__name__)


def parse_package_json(filepath: str) -> Set[str]:
    """Extract dependency package names from package.json."""
    deps = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for section in ["dependencies", "devDependencies", "peerDependencies"]:
                if section in data and isinstance(data[section], dict):
                    deps.update(data[section].keys())
    except Exception as e:
        logger.debug(f"Could not parse {filepath}: {e}")
    return deps


def parse_requirements_txt(filepath: str) -> Set[str]:
    """Extract package names from requirements.txt."""
    deps = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
                    if pkg_name:
                        deps.add(pkg_name)
    except Exception as e:
        logger.debug(f"Could not parse {filepath}: {e}")
    return deps


def parse_pyproject_toml(filepath: str) -> Set[str]:
    """Extract dependency names from pyproject.toml."""
    deps = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # Basic string search for dependencies list
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('"') or line.startswith("'"):
                    clean = line.strip("\", '")
                    pkg = clean.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
                    if pkg and not pkg.startswith("setuptools"):
                        deps.add(pkg)
    except Exception as e:
        logger.debug(f"Could not parse {filepath}: {e}")
    return deps


def get_workspace_file_extensions(workspace_path: str, max_files: int = 100) -> Dict[str, int]:
    """Scan workspace directory for dominant file extensions."""
    counts: Dict[str, int] = {}
    count = 0
    try:
        for root, dirs, files in os.walk(workspace_path):
            # Skip hidden and cache dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["venv", "node_modules", "__pycache__"]]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext:
                    counts[ext] = counts.get(ext, 0) + 1
                count += 1
                if count >= max_files:
                    break
            if count >= max_files:
                break
    except Exception as e:
        logger.debug(f"Error scanning extensions in {workspace_path}: {e}")
    return counts


def build_project_context_text(workspace_path: str = ".") -> str:
    """Build a unified text description of project dependencies and codebase language types."""
    dependencies: Set[str] = set()

    pkg_json = os.path.join(workspace_path, "package.json")
    if os.path.exists(pkg_json):
        dependencies.update(parse_package_json(pkg_json))

    req_txt = os.path.join(workspace_path, "requirements.txt")
    if os.path.exists(req_txt):
        dependencies.update(parse_requirements_txt(req_txt))

    pyproj = os.path.join(workspace_path, "pyproject.toml")
    if os.path.exists(pyproj):
        dependencies.update(parse_pyproject_toml(pyproj))

    ext_counts = get_workspace_file_extensions(workspace_path)
    top_exts = sorted(ext_counts.keys(), key=lambda k: ext_counts[k], reverse=True)[:5]

    parts = []
    if dependencies:
        parts.append(f"Dependencies: {', '.join(sorted(list(dependencies)))}")
    if top_exts:
        parts.append(f"File Types: {', '.join(top_exts)}")

    context_str = " | ".join(parts) if parts else "Generic Software Project"
    return f"Project Context: {context_str}"


def get_project_context_embedding(context_text: str) -> List[float]:
    """Generate vector embedding for project context using the same all-MiniLM-L6-v2 model."""
    model = get_embedding_model()
    vector = model.encode([context_text])[0].tolist()
    return vector
