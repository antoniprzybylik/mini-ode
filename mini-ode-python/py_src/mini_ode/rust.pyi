import torch
from typing import Final, Optional, Sequence, Tuple

__version__: str

class Optimizer:
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Solver:
    def solve(
        self,
        f: torch.jit.ScriptFunction,
        interval: Sequence[float],
        y0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Newton(Optimizer):
    def __init__(
        self,
        max_steps: int,
        gtol: Optional[float] = None,
        ftol: Optional[float] = None,
    ) -> None: ...

class Halley(Optimizer):
    def __init__(
        self,
        max_steps: int,
        gtol: Optional[float] = None,
        ftol: Optional[float] = None,
    ) -> None: ...

class BFGS(Optimizer):
    def __init__(
        self,
        max_steps: int,
        gtol: Optional[float] = None,
        ftol: Optional[float] = None,
    ) -> None: ...

class CG(Optimizer):
    def __init__(
        self,
        max_steps: int,
        gtol: Optional[float] = None,
        ftol: Optional[float] = None,
    ) -> None: ...

class EulerMethodSolver(Solver):
    step: Final[float]
    def __init__(self, step: float) -> None: ...

class RK4MethodSolver(Solver):
    step: Final[float]
    def __init__(self, step: float) -> None: ...

class RKF45MethodSolver(Solver):
    rtol: Final[float]
    atol: Final[float]
    min_step: Final[float]
    safety_factor: Final[float]
    def __init__(
        self,
        rtol: float,
        atol: float,
        min_step: float,
        safety_factor: float,
    ) -> None: ...

class ImplicitEulerMethodSolver(Solver):
    step: Final[float]
    optimizer: Final[Optimizer]
    def __init__(
        self,
        step: float,
        optimizer: Optimizer,
    ) -> None: ...

class GLRK4MethodSolver(Solver):
    step: Final[float]
    optimizer: Final[Optimizer]
    def __init__(
        self,
        step: float,
        optimizer: Optimizer,
    ) -> None: ...

class ROW1MethodSolver(Solver):
    step: Final[float]
    def __init__(self, step: float) -> None: ...
