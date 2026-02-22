"""Repository guardrails for gateway-only strategy order routing."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "strategies"
BANNED_BROKER_ORDER_METHODS = {
    "submit_order",
    "submit_order_advanced",
    "place_order",
    "close_position",
    "liquidate_position",
    "liquidate_all",
    "cancel_order",
}


@dataclass
class StrategyClassInfo:
    """Minimal AST metadata used to enforce constructor and routing rules."""

    name: str
    path: Path
    class_node: ast.ClassDef
    base_names: list[str]


def _read_strategy_classes() -> dict[str, StrategyClassInfo]:
    """Parse strategy modules and return class metadata by class name."""
    class_map: dict[str, StrategyClassInfo] = {}
    for path in sorted(STRATEGY_DIR.glob("*.py")):
        if path.name in {"__init__.py", "base_strategy.py"}:
            continue

        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            base_names: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)

            class_map[node.name] = StrategyClassInfo(
                name=node.name,
                path=path,
                class_node=node,
                base_names=base_names,
            )

    return class_map


def _inherits_base_strategy(
    class_name: str,
    class_map: dict[str, StrategyClassInfo],
    stack: set[str] | None = None,
) -> bool:
    """Resolve transitive inheritance to determine BaseStrategy lineage."""
    if class_name == "BaseStrategy":
        return True
    if class_name not in class_map:
        return False

    stack = stack or set()
    if class_name in stack:
        return False
    stack.add(class_name)

    info = class_map[class_name]
    for base_name in info.base_names:
        if base_name == "BaseStrategy":
            return True
        if _inherits_base_strategy(base_name, class_map, stack):
            return True
    return False


def _live_strategy_classes() -> list[StrategyClassInfo]:
    """Return strategy classes that can run live (exclude backtest-only classes)."""
    class_map = _read_strategy_classes()
    live_classes: list[StrategyClassInfo] = []

    for info in class_map.values():
        if info.name.endswith("Backtest"):
            continue
        if info.path.stem.endswith("_backtest"):
            continue
        if _inherits_base_strategy(info.name, class_map):
            live_classes.append(info)

    return sorted(live_classes, key=lambda x: (x.path.name, x.name))


def test_live_strategies_expose_order_gateway_constructor():
    """Every live strategy constructor must accept order_gateway wiring."""
    missing: list[str] = []
    for info in _live_strategy_classes():
        init_method = next(
            (
                node
                for node in info.class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            ),
            None,
        )

        # Inherited constructors are acceptable (they resolve to BaseStrategy path).
        if init_method is None:
            continue

        arg_names = [arg.arg for arg in init_method.args.args]
        has_kwargs = init_method.args.kwarg is not None
        if "order_gateway" not in arg_names and not has_kwargs:
            missing.append(f"{info.path.name}:{info.name}")

    assert (
        not missing
    ), "Live strategy constructors missing gateway wiring support:\n- " + "\n- ".join(missing)


def test_live_strategies_do_not_submit_orders_directly_to_broker():
    """Live strategy classes must route orders through BaseStrategy helpers/gateway."""
    violations: list[str] = []
    for info in _live_strategy_classes():
        for node in ast.walk(info.class_node):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in BANNED_BROKER_ORDER_METHODS:
                continue

            value = node.func.value
            if not isinstance(value, ast.Attribute):
                continue
            if value.attr != "broker":
                continue
            if not isinstance(value.value, ast.Name) or value.value.id != "self":
                continue

            violations.append(
                f"{info.path.name}:{info.name}:{node.lineno} uses self.broker.{node.func.attr}()"
            )

    assert (
        not violations
    ), "Direct broker order submission found in live strategy classes:\n- " + "\n- ".join(
        violations
    )
