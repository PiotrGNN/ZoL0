#!/usr/bin/env python3
"""
Auto-generate dependency DAG and domain map for the monorepo.
Outputs PlantUML and Graphviz files for docs/DOMAIN_MAP.md.
"""
import os
import ast
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIRS = [
    os.path.join(REPO_ROOT, "core"),
    os.path.join(REPO_ROOT, "data"),
    os.path.join(REPO_ROOT, "python_libs"),
    os.path.join(REPO_ROOT, "ZoL0"),
    os.path.join(REPO_ROOT, "Zolo"),
]


def find_py_files(base_dirs):
    for base in base_dirs:
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(root, f)


def parse_imports(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except Exception:
            return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def build_dependency_graph():
    graph = defaultdict(set)
    for pyfile in find_py_files(SRC_DIRS):
        mod = os.path.relpath(pyfile, REPO_ROOT).replace(os.sep, ".")[:-3]
        for dep in parse_imports(pyfile):
            graph[mod].add(dep)
    return graph


def write_plantuml(graph, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("@startuml\n")
        for mod, deps in graph.items():
            for dep in deps:
                f.write(f'"{mod}" --> "{dep}"\n')
        f.write("@enduml\n")


def write_graphviz(graph, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("digraph dependencies {\n")
        for mod, deps in graph.items():
            for dep in deps:
                f.write(f'  "{mod}" -> "{dep}";\n')
        f.write("}\n")


def main():
    graph = build_dependency_graph()
    write_plantuml(graph, os.path.join(REPO_ROOT, "docs", "dependency_dag.puml"))
    write_graphviz(graph, os.path.join(REPO_ROOT, "docs", "dependency_dag.dot"))
    print("Dependency DAG generated: docs/dependency_dag.puml, docs/dependency_dag.dot")


if __name__ == "__main__":
    main()
