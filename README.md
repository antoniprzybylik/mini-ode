# A minimalistic ODE solvers library built on top of PyTorch

|Solver               |Method             |Purpose                                                                      |
|-----------------------------------------------------------------------------------------------------------------------|
|solve_euler          |Euler method                                   |Simple and efficient method                      |
|solve_rk4            |Fourth Order Runge Kutta method                |General purpose solver with fixed step length    |
|solve_implicit_euler |Implicit Euler method                          |Ill-conditioned problems                         |
|solve_glrk4          |Fourth Order Gauss-Legendre-Runge-Kutta method |Precise solver for ill-conditioned problems      |
|solve_rkf45          |Runge-Kutta-Fehlberg 4(5) method               |General purpose solver with adaptive step length |
|solve_row1           |First Order Rosenbrock-Wanner method           |More efficient simplified Implicit Euler method  |
