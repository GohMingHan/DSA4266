"""Stable source fingerprints matching the unchanged Python 3.12 golden fixture.

ast.dump is a debug display, not a stable serialization: Python 3.13 hides
empty lists by default. This encoder always includes lists, freezes optional
None-field handling, and supplies the empty type_params field absent in 3.11.
It never discards executable fields, nonempty type parameters or unknown fields.
No model/dataset imports and no runtime use outside tests.
"""
import ast
import copy
import hashlib


# Optional None fields in the reference AST schema (not Constant.value=None).
# Explicit names avoid depending on version-specific AST class defaults.
_OPTIONAL_NONE = {
    'FunctionDef': {'returns','type_comment'},
    'AsyncFunctionDef': {'returns','type_comment'},
    'Assign': {'type_comment'}, 'AnnAssign': {'value'},
    'For': {'type_comment'}, 'AsyncFor': {'type_comment'},
    'With': {'type_comment'}, 'AsyncWith': {'type_comment'},
    'Return': {'value'}, 'Yield': {'value'}, 'Raise': {'exc','cause'},
    'Assert': {'msg'}, 'ExceptHandler': {'type','name'},
    'ImportFrom': {'module'}, 'arguments': {'vararg','kwarg'},
    'arg': {'annotation','type_comment'}, 'keyword': {'arg'},
    'alias': {'asname'}, 'withitem': {'optional_vars'},
    'Constant': {'kind'}, 'FormattedValue': {'format_spec'},
    'Slice': {'lower','upper','step'}, 'match_case': {'guard'},
    'MatchAs': {'pattern','name'}, 'MatchStar': {'name'},
    'MatchMapping': {'rest'}, 'TypeVar': {'bound'},
}
_DEFINITIONS = {'FunctionDef','AsyncFunctionDef','ClassDef'}


def stable_ast_dump(value):
    """Single-line structural encoding; deliberately independent of ast.dump."""
    if isinstance(value,list):
        return '['+', '.join(stable_ast_dump(item) for item in value)+']'
    if not isinstance(value,ast.AST):
        return repr(value)
    kind=type(value).__name__
    fields=list(ast.iter_fields(value))
    if kind in _DEFINITIONS and not any(name=='type_params' for name,_ in fields):
        fields.append(('type_params',[]))
    encoded=[]
    for name,item in fields:
        if item is None and name in _OPTIONAL_NONE.get(kind,()):
            continue
        encoded.append(name+'='+stable_ast_dump(item))
    return kind+'('+', '.join(encoded)+')'


class StripProjectImports(ast.NodeTransformer):
    """Same narrowly defined import-location exception as the original test."""
    def visit_ImportFrom(self,node):
        if node.level or node.module and (node.module=='src' or node.module.startswith('src.')):
            return None
        return node


def definition_fingerprint(node):
    canonical=StripProjectImports().visit(copy.deepcopy(node))
    return hashlib.sha256(stable_ast_dump(canonical).encode('utf-8')).hexdigest()
