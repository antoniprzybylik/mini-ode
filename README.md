# A minimalistic ODE solvers library built on top of PyTorch

| **Solver**            | **Method**                                     | **Purpose / Use Case**                                       | **Implicit** | **Adaptive step** |
|------------------------|-----------------------------------------------|---------------------------------------------------------------|--------------|-------------------|
| `solve_euler`          | Euler method                                  | Simple and fast, good for teaching and very basic problems.   | `no`         | `no`              |
| `solve_rk4`            | Fourth Order Runge-Kutta (RK4)                | Widely used general-purpose solver with fixed step size.      | `no`         | `no`              |
| `solve_implicit_euler` | Implicit Euler method                         | Suitable for stiff or ill-conditioned problems.               | `yes`        | `no`              |
| `solve_glrk4`          | Gauss-Legendre Runge-Kutta (Order 4)          | High-precision solver for stiff or ill-conditioned problems.  | `yes`        | `no`              |
| `solve_rkf45`          | Runge-Kutta-Fehlberg 4(5)                     | Adaptive step size, balances efficiency and accuracy.         | `no`         | `yes`             |
| `solve_row1`           | First Order Rosenbrock-Wanner (ROW1) Method   | Efficient alternative to Implicit Euler for stiff systems.    | `semi`       | `no`              |
