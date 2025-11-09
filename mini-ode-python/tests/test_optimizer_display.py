import mini_ode

def test_optimizer_display_cg_case1():
    optimizer = mini_ode.optimizers.CG(10)
    assert optimizer.__str__() == "CG(max_steps=10)"
    assert optimizer.__repr__() == "CG(max_steps=10)"

def test_optimizer_display_cg_case2():
    optimizer = mini_ode.optimizers.CG(10, gtol=2.1)
    assert optimizer.__str__() == "CG(max_steps=10, gtol=2.1)"
    assert optimizer.__repr__() == "CG(max_steps=10, gtol=2.1)"

def test_optimizer_display_cg_case3():
    optimizer = mini_ode.optimizers.CG(10, ftol=1.1)
    assert optimizer.__str__() == "CG(max_steps=10, ftol=1.1)"
    assert optimizer.__repr__() == "CG(max_steps=10, ftol=1.1)"

def test_optimizer_display_cg_case4():
    optimizer = mini_ode.optimizers.CG(10, gtol=4.5, ftol=3.1)
    assert optimizer.__str__() == "CG(max_steps=10, gtol=4.5, ftol=3.1)"
    assert optimizer.__repr__() == "CG(max_steps=10, gtol=4.5, ftol=3.1)"

def test_optimizer_display_newton_case1():
    optimizer = mini_ode.optimizers.Newton(10)
    assert optimizer.__str__() == "Newton(max_steps=10)"
    assert optimizer.__repr__() == "Newton(max_steps=10)"

def test_optimizer_display_newton_case2():
    optimizer = mini_ode.optimizers.Newton(10, gtol=2.1)
    assert optimizer.__str__() == "Newton(max_steps=10, gtol=2.1)"
    assert optimizer.__repr__() == "Newton(max_steps=10, gtol=2.1)"

def test_optimizer_display_newton_case3():
    optimizer = mini_ode.optimizers.Newton(10, ftol=1.1)
    assert optimizer.__str__() == "Newton(max_steps=10, ftol=1.1)"
    assert optimizer.__repr__() == "Newton(max_steps=10, ftol=1.1)"

def test_optimizer_display_newton_case4():
    optimizer = mini_ode.optimizers.Newton(10, gtol=4.5, ftol=3.1)
    assert optimizer.__str__() == "Newton(max_steps=10, gtol=4.5, ftol=3.1)"
    assert optimizer.__repr__() == "Newton(max_steps=10, gtol=4.5, ftol=3.1)"

def test_optimizer_display_halley_case1():
    optimizer = mini_ode.optimizers.Halley(10)
    assert optimizer.__str__() == "Halley(max_steps=10)"
    assert optimizer.__repr__() == "Halley(max_steps=10)"

def test_optimizer_display_halley_case2():
    optimizer = mini_ode.optimizers.Halley(10, gtol=2.1)
    assert optimizer.__str__() == "Halley(max_steps=10, gtol=2.1)"
    assert optimizer.__repr__() == "Halley(max_steps=10, gtol=2.1)"

def test_optimizer_display_halley_case3():
    optimizer = mini_ode.optimizers.Halley(10, ftol=1.1)
    assert optimizer.__str__() == "Halley(max_steps=10, ftol=1.1)"
    assert optimizer.__repr__() == "Halley(max_steps=10, ftol=1.1)"

def test_optimizer_display_halley_case4():
    optimizer = mini_ode.optimizers.Halley(10, gtol=4.5, ftol=3.1)
    assert optimizer.__str__() == "Halley(max_steps=10, gtol=4.5, ftol=3.1)"
    assert optimizer.__repr__() == "Halley(max_steps=10, gtol=4.5, ftol=3.1)"

def test_optimizer_display_bfgs_case1():
    optimizer = mini_ode.optimizers.BFGS(10)
    assert optimizer.__str__() == "BFGS(max_steps=10)"
    assert optimizer.__repr__() == "BFGS(max_steps=10)"

def test_optimizer_display_bfgs_case2():
    optimizer = mini_ode.optimizers.BFGS(10, gtol=2.1)
    assert optimizer.__str__() == "BFGS(max_steps=10, gtol=2.1)"
    assert optimizer.__repr__() == "BFGS(max_steps=10, gtol=2.1)"

def test_optimizer_display_bfgs_case3():
    optimizer = mini_ode.optimizers.BFGS(10, ftol=1.1)
    assert optimizer.__str__() == "BFGS(max_steps=10, ftol=1.1)"
    assert optimizer.__repr__() == "BFGS(max_steps=10, ftol=1.1)"

def test_optimizer_display_bfgs_case4():
    optimizer = mini_ode.optimizers.BFGS(10, gtol=4.5, ftol=3.1)
    assert optimizer.__str__() == "BFGS(max_steps=10, gtol=4.5, ftol=3.1)"
    assert optimizer.__repr__() == "BFGS(max_steps=10, gtol=4.5, ftol=3.1)"
