from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    num_add = f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
    num_sub = f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    denominator = 2 * epsilon
    return (num_add - num_sub) / denominator


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = []
    result = []

    def DFSTraverse(node: Variable) -> None:
        if node.is_constant() or node.unique_id in visited:
            return
        if not node.is_leaf():
            for parent in node.parents:
                DFSTraverse(parent)
        visited.append(node.unique_id)
        result.append(node)

    DFSTraverse(variable)
    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    dict = {variable.unique_id: deriv}
    for var in topological_sort(variable=variable):
        derivative = dict.get(var.unique_id)
        if var.is_leaf():
            var.accumulate_derivative(derivative)
        else:
            for back_var, back_deriv in var.chain_rule(derivative):
                dict.setdefault(back_var.unique_id, 0.0)
                dict[back_var.unique_id] = dict[back_var.unique_id] + back_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
