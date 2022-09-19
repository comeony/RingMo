from typing import Callable

from mindspore import nn


def named_apply(fn: Callable, module: nn.Cell, name='', depth_first=True, include_root=False) -> nn.Cell:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module._children_scope_recursive():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
