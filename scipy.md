**Common Options:**

[disp, maxiter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)

**Method-specific Options:**

| Method | Options |
|--------|---------|
|[Nelder-Mead](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)|maxfev, xatol, fatol, adaptive|
|[Powell](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html)|maxfev, xtol, ftol|
|[CG](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html)|gtol, norm, eps, finite_diff_rel_step, c1, c2|
|[BFGS](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html)|gtol, norm, eps, finite_diff_rel_step, xrtol, c1, c2|
|[Newton-CG](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html)|xtol, eps, c1, c2|
|[L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)^1^|*disp*, maxcor, ftol, gtol, eps, maxfun, iprint, maxls, finite_diff_rel_step|
|[TNC](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html)^2^|maxfun, eps, scale, offset, maxCGit, eta, stepmx, accuracy, minfev, ftol, xtol, gtol, rescale, finite_diff_rel_step, ~~maxiter~~|
|[COBYLA](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html)|rhobeg, tol, catol|
|[COBYQA](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html)|maxfev, f_target, feasibility_tol, initial_tr_radius, final_tr_radius, scale|
|[SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)|ftol, eps, finite_diff_rel_step|
|[trust-constr](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html)|gtol, xtol, barrier_tol, sparse_jacobian, initial_tr_radius, initial_constr_penalty, initial_barrier_parameter, initial_barrier_tolerance, factorization_method, finite_diff_rel_step, verbose|
|[differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)|strategy, popsize, tol, mutation, recombination, rng, polish, init, atol, updating|

**Notes:**

1. Options in *italics* override a common option with a different type or behavior.
2. Options with ~~strikethrough~~ indicate a common option that is not supported.
