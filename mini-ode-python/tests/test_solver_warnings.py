import pytest
import warnings
import torch
import mini_ode


class TestSolverWarnings:
    """Test that ODE solver warnings are properly issued and can be controlled."""
    
    def test_euler_divergence_warning_emitted(self):
        """Verify that divergence warning is issued when solution explodes."""
        
        def f(x, y):
            return (y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)) * 0.45
        traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))
        
        solver = mini_ode.EulerMethodSolver(step=1.)
        
        with pytest.warns(RuntimeWarning, match="solution norm exceeded"):
            xs, ys = solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
    
    def test_warnings_can_be_muted_simplefilter(self):
        """Verify warnings can be muted using warnings.simplefilter."""
        
        def f(x, y):
            return (y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)) * 0.45
        traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))
        
        solver = mini_ode.EulerMethodSolver(step=1.)
        
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore", RuntimeWarning)
            
            xs, ys = solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
            
            # When ignored, warnings should NOT be captured at all
            runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0
    
    def test_warnings_suppressed_no_record(self):
        """Correct approach: when filtered as ignore, warnings don't reach record."""
        
        def f(x, y):
            return (y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)) * 0.45
        traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))
        
        solver = mini_ode.EulerMethodSolver(step=1.)
        
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("always", category=UserWarning)  # Record other warnings
            
            solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
            
            # RuntimeWarning should be suppressed (filtered out before recording)
            runtime_ws = [w for w in ws if issubclass(w.category, RuntimeWarning)]
            assert len(runtime_ws) == 0
    
    def test_warning_normal_behavior_no_filter(self):
        """Verify warning IS emitted when no filters are applied."""
        
        def f(x, y):
            return (y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)) * 0.45
        traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))
        
        solver = mini_ode.EulerMethodSolver(step=1.)
        
        # No special filters - should emit warning
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            
            solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
            
            runtime_ws = [w for w in ws if issubclass(w.category, RuntimeWarning)]
            assert len(runtime_ws) == 1
            assert "solution norm exceeded" in str(runtime_ws[0].message)
    
    def test_warning_ignore_vs_always_comparison(self):
        """Demonstrate difference between 'ignore' and 'always' filters."""
        
        def f(x, y):
            return (y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)) * 0.45
        traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))
        
        solver = mini_ode.EulerMethodSolver(step=1.)
        
        # Count with 'always'
        with warnings.catch_warnings(record=True) as ws_always:
            warnings.simplefilter("always")
            solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
            always_count = len([w for w in ws_always if issubclass(w.category, RuntimeWarning)])
        
        # Count with 'ignore'
        with warnings.catch_warnings(record=True) as ws_ignore:
            warnings.simplefilter("ignore", RuntimeWarning)
            solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
            ignore_count = len([w for w in ws_ignore if issubclass(w.category, RuntimeWarning)])
        
        assert always_count == 1
        assert ignore_count == 0
    
    def test_warning_once_policy(self):
        """Test 'once' policy - warning issued only once per location."""
        
        def f(x, y):
            return (y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)) * 0.45
        traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))
        
        solver = mini_ode.EulerMethodSolver(step=1.)
        
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("once")
            
            # Call multiple times
            for _ in range(3):
                try:
                    solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
                except Exception:
                    pass
            
            # Only one warning should be recorded due to 'once' policy
            runtime_ws = [w for w in ws if issubclass(w.category, RuntimeWarning)]
            assert len(runtime_ws) == 1
    
    def test_warning_default_action(self):
        """Test default warning behavior."""
        
        def f(x, y):
            return (y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)) * 0.45
        traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))
        
        solver = mini_ode.EulerMethodSolver(step=1.)
        
        # Reset to default and verify warning is emitted
        with warnings.catch_warnings(record=True) as ws:
            warnings.resetwarnings()
            warnings.filterwarnings("default", category=RuntimeWarning)
            
            solver.solve(traced_f, (0., 15.), torch.tensor([1.5, 0.0]))
            
            runtime_ws = [w for w in ws if issubclass(w.category, RuntimeWarning)]
            assert len(runtime_ws) == 1
