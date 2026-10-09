"""Dependency plans compiled from our own feature formulas, never user code.

This keeps the group implementation as the formula source of truth. Configuration
loops and literal dictionaries become graph metadata. Numeric expressions are
evaluated only when their output is requested. Stateful loops remain atomic
kernels; shared intermediate results are cached for one evaluation only.
"""

import ast
import copy
import inspect
import textwrap
from dataclasses import dataclass


class DynamicExpression(Exception):
    """An expression depends on input data, rather than configuration."""


@dataclass(eq=False)
class Node:
    evaluator: object
    history: int | None = 0
    dependencies: tuple = ()

    def get(self, context):
        if self not in context:
            context[self] = self.evaluator(context)
        value = context[self]
        remaining = context.get("remaining")
        if remaining is not None:
            remaining[self] -= 1
            if remaining[self] == 0:
                del context[self]
        return value


def constant(value):
    return Node(lambda context: value)


def input_node(name):
    return Node(lambda context: context["inputs"][name])


class References(ast.NodeTransformer):
    def __init__(self, planner):
        self.planner = planner
        self.nodes = []
        self.bound = set()

    def reference(self, node):
        self.nodes.append(node)
        return ast.Call(ast.Name("_value", ast.Load()), [ast.Constant(len(self.nodes) - 1)], [])

    def visit_Name(self, node):
        if (
            isinstance(node.ctx, ast.Load)
            and node.id not in self.bound
            and node.id in self.planner.env
        ):
            value = self.planner.env[node.id]
            if isinstance(value, dict):
                raise RuntimeError(f"Unspecialized dictionary reference: {node.id}")
            return self.reference(value if isinstance(value, Node) else constant(value))
        return node

    def visit_Subscript(self, node):
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "regression"
        ):
            # The existing callback returns four descriptors. Request only one.
            from pfe.features._kernels import regression_component

            self.planner.globals["_regression_component"] = regression_component
            return self.visit(
                ast.Call(
                    ast.Name("_regression_component", ast.Load()),
                    [*node.value.args, node.slice],
                    [],
                )
            )
        if isinstance(node.value, ast.Name) and node.value.id in self.planner.env:
            mapping = self.planner.env[node.value.id]
            if isinstance(mapping, dict):
                key = self.planner.static(node.slice)
                return self.reference(mapping[key])
        return self.generic_visit(node)

    def visit_Lambda(self, node):
        node.args.defaults = [self.visit(value) for value in node.args.defaults]
        old = self.bound.copy()
        self.bound.update(
            arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        )
        node.body = self.visit(node.body)
        self.bound = old
        return node

    def comprehension(self, node):
        old = self.bound.copy()
        for generator in node.generators:
            generator.iter = self.visit(generator.iter)
            self.bound.update(n.id for n in ast.walk(generator.target) if isinstance(n, ast.Name))
            generator.ifs = [self.visit(value) for value in generator.ifs]
        for name in ("elt", "key", "value"):
            if hasattr(node, name):
                setattr(node, name, self.visit(getattr(node, name)))
        self.bound = old
        return node

    visit_GeneratorExp = comprehension
    visit_ListComp = comprehension
    visit_SetComp = comprehension
    visit_DictComp = comprehension


class Planner:
    def __init__(self, function, configuration, *, inputs=("bars",), overrides=None):
        self.globals = {**function.__globals__, **(overrides or {})}
        self.env = {**configuration, **{name: input_node(name) for name in inputs}}
        self.output = {}
        self.put = None
        self._counts = {}
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        self.block(tree.body[0].body)

    def static(self, expression):
        if isinstance(expression, ast.Name):
            value = self.env.get(expression.id)
            if value is not None and not isinstance(value, Node):
                return value
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute):
            if expression.func.attr in {"items", "keys", "values"}:
                obj = self.static(expression.func.value)
                return list(getattr(obj, expression.func.attr)())
        # Configuration evaluation must never touch a data node.
        names = {n.id for n in ast.walk(expression) if isinstance(n, ast.Name)}
        if any(isinstance(self.env.get(name), Node) for name in names):
            raise DynamicExpression
        namespace = {**self.globals, **self.env}
        try:
            tree = ast.fix_missing_locations(ast.Expression(copy.deepcopy(expression)))
            return eval(compile(tree, "<pfe-config>", "eval"), namespace)
        except (NameError, TypeError, AttributeError) as exc:
            raise DynamicExpression from exc

    def expression(self, expression):
        if isinstance(expression, ast.Dict):
            result = {}
            for key, value in zip(expression.keys, expression.values, strict=True):
                if key is None:
                    result.update(self.mapping(value))
                else:
                    result[self.static(key)] = self.expression(value)
            return result
        if isinstance(expression, ast.DictComp):
            generator = expression.generators[0]
            result = {}
            iterable = (
                ("open", "high", "low", "close", "volume")
                if isinstance(generator.iter, ast.Name) and generator.iter.id == "bars"
                else self.static(generator.iter)
            )
            for value in iterable:
                self.bind(generator.target, value)
                result[self.static(expression.key)] = self.expression(expression.value)
            return result
        if isinstance(expression, ast.Name) and expression.id in self.env:
            value = self.env[expression.id]
            return value if isinstance(value, (Node, dict)) else constant(value)
        refs = References(self)
        transformed = refs.visit(copy.deepcopy(expression))
        if isinstance(transformed, ast.GeneratorExp):
            transformed = ast.Call(ast.Name("tuple", ast.Load()), [transformed], [])
        code = compile(
            ast.fix_missing_locations(ast.Expression(transformed)), "<pfe-feature>", "eval"
        )

        def evaluate(context):
            # Callbacks may use the same reference on every row. Resolve it once.
            resolved = {}

            def value(i):
                if i not in resolved:
                    resolved[i] = refs.nodes[i].get(context)
                return resolved[i]

            namespace = {**self.globals, "_value": value}
            return eval(code, namespace)

        history = max((node.history or 0 for node in refs.nodes), default=0)
        if any(node.history is None for node in refs.nodes):
            history = None
        for call in (node for node in ast.walk(expression) if isinstance(node, ast.Call)):
            function = call.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if name in {
                "ewm",
                "cumsum",
                "cumprod",
                "cummax",
                "cummin",
                "expanding",
                "ffill",
                "event_age",
                "adaptive_mean",
                "groupby",
                "ATR",
            }:
                history = None
            elif history is not None and name in {"rolling", "shift", "diff", "pct_change"}:
                argument = call.args[0] if call.args else ast.Constant(1)
                try:
                    size = int(self.static(argument))
                    history += max(0, size - 1) if name == "rolling" else size
                except (DynamicExpression, TypeError, ValueError):
                    history = None
            elif history is not None and name in {"wma", "rolling_kernel"}:
                try:
                    history += max(0, int(self.static(call.args[1])) - 1)
                except (DynamicExpression, TypeError, ValueError):
                    history = None
        return Node(evaluate, history, tuple(refs.nodes))

    def mapping(self, expression):
        if isinstance(expression, ast.Name) and isinstance(self.env.get(expression.id), dict):
            return self.env[expression.id]
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            if expression.func.id == "dict":
                return {item.arg: self.expression(item.value) for item in expression.keywords}
            if expression.func.id in {"regression", "excursion_features", "profile_features"}:
                function = self.globals[expression.func.id]
                parameters = list(inspect.signature(function).parameters)
                bindings = {}
                for name, value in zip(parameters, expression.args, strict=False):
                    try:
                        bindings[name] = self.static(value)
                    except DynamicExpression:
                        bindings[name] = self.expression(value)
                for item in expression.keywords:
                    try:
                        bindings[item.arg] = self.static(item.value)
                    except DynamicExpression:
                        bindings[item.arg] = self.expression(item.value)
                return Planner(function, bindings, inputs=()).output
        value = self.expression(expression)
        if not isinstance(value, dict):
            raise RuntimeError("Output mappings must be specialized before evaluation")
        return value

    def bind(self, target, value):
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            if target.id == "out":
                self.output = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, name in enumerate(target.elts):
                if isinstance(value, Node):
                    item = Node(
                        lambda context, i=i, parent=value: parent.get(context)[i],
                        value.history,
                        (value,),
                    )
                else:
                    item = value[i]
                self.bind(name, item)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            mapping = self.env.get(target.value.id)
            if isinstance(mapping, dict):
                mapping[self.static(target.slice)] = (
                    value if isinstance(value, Node) else constant(value)
                )
            else:
                raise DynamicExpression
        else:
            raise DynamicExpression

    def atomic(self, statement):
        """Stateful numeric loops execute as one task, not once per output."""
        loaded = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)}
        bindings = {name: self.env[name] for name in loaded if name in self.env}
        if any(isinstance(value, dict) for value in bindings.values()):
            raise RuntimeError("Stateful kernel cannot mutate an output dictionary")
        writes = {
            n.id
            for n in ast.walk(statement)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        for node in ast.walk(statement):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for member in ast.walk(target):
                        if isinstance(member, (ast.Subscript, ast.Attribute)):
                            while isinstance(member, (ast.Subscript, ast.Attribute)):
                                member = member.value
                            if isinstance(member, ast.Name):
                                writes.add(member.id)
        code = compile(
            ast.fix_missing_locations(ast.Module([copy.deepcopy(statement)], [])),
            "<pfe-kernel>",
            "exec",
        )

        kernel = None
        returned = sorted(writes & bindings.keys())
        arguments = sorted(bindings)
        if isinstance(statement, (ast.For, ast.While)):

            class Arrays(ast.NodeTransformer):
                def visit_Attribute(self, node):
                    return (
                        self.visit(node.value) if node.attr == "iloc" else self.generic_visit(node)
                    )

                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name) and node.func.id == "zip":
                        # Kernel slices have equal sizes; Numba zip has no strict keyword.
                        node.keywords = [item for item in node.keywords if item.arg != "strict"]
                    return self.generic_visit(node)

            body = Arrays().visit(copy.deepcopy(statement))
            definition = ast.FunctionDef(
                name="_numeric_kernel",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(name) for name in arguments],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[
                    body,
                    ast.Return(
                        ast.Tuple([ast.Name(name, ast.Load()) for name in returned], ast.Load())
                    ),
                ],
                decorator_list=[],
            )
            namespace = self.globals.copy()
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module([definition], [])),
                    "<pfe-numeric-kernel>",
                    "exec",
                ),
                namespace,
            )

            # Imported lazily: selecting a vectorized/native feature needs no JIT.
            def kernel(values, function=namespace["_numeric_kernel"]):
                nonlocal compiled
                if compiled is None:
                    from numba import njit

                    compiled = njit(nogil=True, fastmath=False)(function)
                converted = [
                    value.to_numpy() if hasattr(value, "to_numpy") else value
                    for value in (values[name] for name in arguments)
                ]
                return dict(zip(returned, compiled(*converted), strict=True))

            compiled = None

        def execute(context):
            namespace = {
                **self.globals,
                **{
                    name: value.get(context) if isinstance(value, Node) else value
                    for name, value in bindings.items()
                },
            }
            if kernel is not None:
                return kernel(namespace)
            exec(code, namespace)
            return namespace

        parent = Node(
            execute,
            None
            if isinstance(statement, (ast.For, ast.While))
            else max(
                (value.history or 0 for value in bindings.values() if isinstance(value, Node)),
                default=0,
            ),
            tuple(value for value in bindings.values() if isinstance(value, Node)),
        )
        for name in returned if kernel is not None else writes:
            self.env[name] = Node(
                lambda context, name=name: parent.get(context)[name], parent.history, (parent,)
            )

    def branch(self, statement):
        predicate = self.expression(statement.test)
        before = self.env.copy()
        self.env = {k: v.copy() if isinstance(v, dict) else v for k, v in before.items()}
        self.block(statement.body)
        yes = self.env
        self.env = {k: v.copy() if isinstance(v, dict) else v for k, v in before.items()}
        self.block(statement.orelse)
        no = self.env

        def merge(a, b):
            if a is b:
                return a
            if isinstance(a, dict) or isinstance(b, dict):
                a, b = a or {}, b or {}
                return {key: merge(a.get(key), b.get(key)) for key in a.keys() | b.keys()}
            a = a if isinstance(a, Node) else constant(a)
            b = b if isinstance(b, Node) else constant(b)
            history = (
                None
                if any(node.history is None for node in (predicate, a, b))
                else max(node.history for node in (predicate, a, b))
            )
            return Node(
                lambda context: (a if predicate.get(context) else b).get(context),
                history,
                (predicate, a, b),
            )

        self.env = {key: merge(yes.get(key), no.get(key)) for key in yes.keys() | no.keys()}
        self.output = self.env.get("out", self.output)

    def block(self, statements):
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Return):
                if isinstance(statement.value, ast.Name) and statement.value.id == "out":
                    self.output = self.env["out"]
                else:
                    self.env["out"] = self.output = self.mapping(statement.value)
                return
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.FunctionDef):
                if statement.name != "put":
                    raise RuntimeError("Only the feature writer can be nested")
                self.put = statement
            elif isinstance(statement, ast.Assign):
                try:
                    # Keep scalar configuration assignments static for loop unrolling.
                    value = self.static(statement.value)
                except DynamicExpression:
                    value = self.expression(statement.value)
                try:
                    for target in statement.targets:
                        self.bind(target, value)
                except DynamicExpression:
                    self.atomic(statement)
            elif isinstance(statement, ast.AugAssign):
                if isinstance(statement.target, ast.Name):
                    value = ast.BinOp(
                        ast.Name(statement.target.id, ast.Load()), statement.op, statement.value
                    )
                    self.bind(statement.target, self.expression(value))
                else:
                    self.atomic(statement)
            elif isinstance(statement, ast.For):
                try:
                    values = (
                        ("open", "high", "low", "close", "volume")
                        if isinstance(statement.iter, ast.Name) and statement.iter.id == "bars"
                        else self.static(statement.iter)
                    )
                except DynamicExpression:
                    # A literal dict's values are graph nodes, not numeric work.
                    if (
                        isinstance(statement.iter, ast.Call)
                        and isinstance(statement.iter.func, ast.Attribute)
                        and isinstance(statement.iter.func.value, ast.Dict)
                    ):
                        mapping = self.mapping(statement.iter.func.value)
                        values = list(getattr(mapping, statement.iter.func.attr)())
                    elif isinstance(statement.iter, (ast.Tuple, ast.List)):

                        def item(expression):
                            if isinstance(expression, (ast.Tuple, ast.List)):
                                return tuple(item(value) for value in expression.elts)  # noqa: B023
                            try:
                                return self.static(expression)
                            except DynamicExpression:
                                return self.expression(expression)

                        values = [item(value) for value in statement.iter.elts]
                    elif (
                        isinstance(statement.iter, ast.Call)
                        and isinstance(statement.iter.func, ast.Attribute)
                        and statement.iter.func.attr == "items"
                    ):
                        values = list(self.mapping(statement.iter.func.value).items())
                    else:
                        self.atomic(statement)
                        continue
                for value in values:
                    self.bind(statement.target, value)
                    self.block(statement.body)
            elif isinstance(statement, ast.If):
                try:
                    condition = self.static(statement.test)
                except DynamicExpression:
                    if any(isinstance(item, ast.Return) for item in statement.body):
                        expanded = ast.If(
                            statement.test,
                            statement.body,
                            [*statement.orelse, *statements[index + 1 :]],
                        )
                        self.branch(expanded)
                        return
                    # In a configuration loop, continue skips its remaining body.
                    if len(statement.body) == 1 and isinstance(statement.body[0], ast.Continue):
                        rest = ast.If(
                            ast.UnaryOp(ast.Not(), statement.test), statements[index + 1 :], []
                        )
                        self.branch(rest)
                        return
                    self.branch(statement)
                else:
                    if (
                        condition
                        and len(statement.body) == 1
                        and isinstance(statement.body[0], ast.Continue)
                    ):
                        return
                    self.block(statement.body if condition else statement.orelse)
            elif isinstance(statement, ast.While):
                self.atomic(statement)
            elif isinstance(statement, ast.Expr):
                call = statement.value
                if isinstance(call, ast.Constant):
                    continue
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "put"
                ):
                    definition = self.put
                    writer_env = self.env.copy()
                    params = definition.args.args
                    arguments = [
                        self.static(arg) if param.arg != "value" else self.expression(arg)
                        for param, arg in zip(params, call.args, strict=False)
                    ]
                    for param, argument in zip(params, arguments, strict=False):
                        self.env[param.arg] = argument
                    defaults = dict(
                        zip(
                            (p.arg for p in params[-len(definition.args.defaults) :]),
                            definition.args.defaults,
                            strict=True,
                        )
                    )
                    for param in params[len(call.args) :]:
                        self.env[param.arg] = self.static(defaults[param.arg])
                    self.block(definition.body)
                    output = self.env["out"]
                    self.env = writer_env
                    self.env["out"] = self.output = output
                elif (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "update"
                ):
                    if call.args:
                        self.env[call.func.value.id].update(self.mapping(call.args[0]))
                    self.env[call.func.value.id].update(
                        {item.arg: self.expression(item.value) for item in call.keywords}
                    )
                else:
                    self.atomic(statement)
            else:
                raise RuntimeError(f"Unsupported formula statement: {ast.dump(statement)}")

    def evaluate(self, names, **inputs):
        names = tuple(names)
        if names not in self._counts:
            counts, visited = {}, set()

            def visit(node):
                counts[node] = counts.get(node, 0) + 1
                if node not in visited:
                    visited.add(node)
                    for dependency in node.dependencies:
                        visit(dependency)

            for name in names:
                visit(self.output[name])
            # Bound metadata growth when callers select many different combinations.
            if len(self._counts) >= 32:
                self._counts.clear()
            self._counts[names] = counts
        context = {"inputs": inputs, "remaining": self._counts[names].copy()}
        return {name: self.output[name].get(context) for name in names}
