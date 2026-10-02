"""Safe user-defined cost formula support.

Parses a math expression string into a compiled callable that works on
both Python scalars and numpy/CuPy arrays (via operator overloading).
"""

import ast
import math

DEFAULT_FORMULA = '(theta_f * F + theta_d * D + theta_m * M) / (theta_s * S)'

ALLOWED_VARIABLES = frozenset({
    'S', 'D', 'F', 'M', 'C', 'n',
    'theta_s', 'theta_d', 'theta_f', 'theta_m',
})

_ALLOWED_FUNC_NAMES = frozenset({
    'abs', 'sqrt', 'log', 'log2', 'log10', 'exp', 'pow',
})


def _safe_func(name, np_name=None):
    """Create a function that dispatches to numpy/cupy based on input type."""
    if np_name is None:
        np_name = name

    def wrapper(*args):
        import numpy as _np
        if hasattr(args[0], '__cuda_array_interface__'):
            import cupy as cp
            return getattr(cp, np_name)(*args)
        if isinstance(args[0], _np.ndarray):
            return getattr(_np, np_name)(*args)
        return getattr(math, name)(*args)

    wrapper.__name__ = name
    return wrapper


_SAFE_NAMESPACE = {
    'abs': _safe_func('fabs', 'abs'),
    'sqrt': _safe_func('sqrt'),
    'log': _safe_func('log'),
    'log2': _safe_func('log2'),
    'log10': _safe_func('log10'),
    'exp': _safe_func('exp'),
    'pow': _safe_func('pow', 'power'),
}


class _SafeFormulaValidator(ast.NodeVisitor):
    """AST visitor that rejects unsafe node types."""

    _ALLOWED_BINOPS = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod,
    )
    _ALLOWED_UNARYOPS = (ast.USub, ast.UAdd)
    _ALLOWED_CMPOPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)
    _ALLOWED_BOOLOPS = (ast.And, ast.Or)

    def visit_Expression(self, node):
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if not isinstance(node.op, self._ALLOWED_BINOPS):
            raise ValueError('Disallowed operator: ' + type(node.op).__name__)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, self._ALLOWED_UNARYOPS):
            raise ValueError('Disallowed unary operator: ' + type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            if not isinstance(op, self._ALLOWED_CMPOPS):
                raise ValueError('Disallowed comparison: ' + type(op).__name__)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        if not isinstance(node.op, self._ALLOWED_BOOLOPS):
            raise ValueError('Disallowed boolean operator: ' + type(node.op).__name__)
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id not in ALLOWED_VARIABLES and node.id not in _ALLOWED_FUNC_NAMES:
            raise ValueError('Unknown variable: ' + node.id
                             + '. Allowed: ' + ', '.join(sorted(ALLOWED_VARIABLES)))

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float)):
            raise ValueError('Only numeric constants allowed, got: ' + repr(node.value))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError('Only simple function calls allowed (no methods or chained calls)')
        if node.func.id not in _ALLOWED_FUNC_NAMES:
            raise ValueError('Unknown function: ' + node.func.id
                             + '. Allowed: ' + ', '.join(sorted(_ALLOWED_FUNC_NAMES)))
        if node.starargs if hasattr(node, 'starargs') else False:
            raise ValueError('*args not allowed in function calls')
        if node.keywords:
            raise ValueError('Keyword arguments not allowed in function calls')
        self.generic_visit(node)

    def generic_visit(self, node):
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
            ast.IfExp, ast.Name, ast.Constant, ast.Call, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd,
            ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
            ast.And, ast.Or,
        )
        if not isinstance(node, allowed):
            raise ValueError('Disallowed expression element: ' + type(node).__name__
                             + '. Only arithmetic, comparisons, and allowed functions are permitted.')
        super().generic_visit(node)


def validate_formula(formula_str):
    """Parse and validate a formula string. Raises ValueError or SyntaxError."""
    if not formula_str or not formula_str.strip():
        raise ValueError('Formula string is empty')
    tree = ast.parse(formula_str.strip(), mode='eval')
    _SafeFormulaValidator().visit(tree)


def compile_cost_formula(formula_str):
    """Compile a formula string into a callable.

    Returns a function with signature:
        f(S, D, F, M, C, n, theta_s, theta_d, theta_f, theta_m) -> cost
    Works on scalars, numpy arrays, and CuPy arrays.
    """
    formula_str = formula_str.strip()
    validate_formula(formula_str)
    code = compile(ast.parse(formula_str, mode='eval'), '<cost_formula>', 'eval')

    # CuPy's kernel JIT uses __import__ internally during array ops,
    # so we allow only that builtin rather than a fully empty dict.
    _restricted_builtins = {'__import__': __builtins__['__import__']} if isinstance(
        __builtins__, dict) else {'__import__': __builtins__.__import__}

    def cost_fn(S, D, F, M, C, n, theta_s, theta_d, theta_f, theta_m):
        return eval(code, {'__builtins__': _restricted_builtins}, {
            'S': S, 'D': D, 'F': F, 'M': M, 'C': C, 'n': n,
            'theta_s': theta_s, 'theta_d': theta_d,
            'theta_f': theta_f, 'theta_m': theta_m,
            **_SAFE_NAMESPACE,
        })

    return cost_fn
