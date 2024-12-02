kolmog: 
    - Test on Kolmogorov flow, orb05
kolmog1:
    - Same structure of pyRB/ra1e6_256/2000a rewritten (not guaranteed that are equivalent)
    - Found bug in update of mu
    - Slow convergence respect to orbs_hook
kolmog2:
    - Only fixed update of mu
    - Better convergence but not as good as orbs_hook
kolmog3:
    - Attempt to replicate orbs_hook. Changed pmN.c = 0, mu0 = 1e-3, mu*=2 instead of 1.2.
kolmog4:
    - Same as kolmog3, but now performs inc_proj over dU instead of U+dU, and doesn't perform project in GMRes perturbation 
    (now it is the same as orbs_hook i think)
