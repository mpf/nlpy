"""
This module implements a purely primal-dual interior-point methods for
bound-constrained optimization. The method uses the primal-dual merit function
of Forsgren and Gill and solves subproblems by means of a truncated conjugate
gradient method with trust region.

References:

 [1] A. Forsgren and Ph. E. Gill, Primal-Dual Interior Methods for Nonconvex
     Nonlinear Programming, SIAM Journal on Optimization, 8(4):1132-1152, 1998

 [2] Ph. E. Gill and E. M. Gertz, A primal-dual trust region algorithm for
     nonlinear optimization, Mathematical Programming, 100(1):49-94, 2004

 [3] P. Armand and D. Orban, A Trust-Region Interior-Point Method for
     Bound-Constrained Programming Based on a Primal-Dual Merit Function,
     Cahier du GERAD G-xxxx, GERAD, Montreal, Quebec, Canada, 2008.

D. Orban, Montreal
"""

from nlpy.model        import NLPModel
from pysparse.sparse   import PysparseMatrix as sp
from nlpy.tools        import List
from nlpy.tools.timing import cputime
from nlpy.tools.norms  import norm2, norm_infty
from math              import sqrt

import numpy as np
import sys

import pdb


class PrimalDualMeritFunction(NLPModel):

    def __init__(self, nlp, **kwargs):

        if nlp.nlowerC + nlp.nupperC + nlp.nrangeC > 0:
            raise ValueError, 'Only bound-constrained problems are supported.'

        if nlp.nfixedB > 0:
            raise ValueError, 'Fixed variables are currently not supported.'

        self.nlp = nlp
        self.mu = kwargs.get('mu', 1.0)

        # Shortcuts for convenience.
        self.lowerB = np.array(nlp.lowerB)
        self.upperB = np.array(nlp.upperB)
        self.rangeB = np.array(nlp.rangeB)
        self.rlB = np.arange(nlp.nlowerB)
        self.ruB = nlp.nlowerB + np.arange(nlp.nupperB)
        self.rrB = nlp.nlowerB + nlp.nupperB + np.arange(nlp.nrangeB)

        # Number of dual variables
        self.nz = nlp.nlowerB + nlp.nupperB + 2 * nlp.nrangeB

        # Total number of variables.
        self.n = self.nlp.n + self.nz

        # The multipliers z associated to bound constraints are ordered as
        # follows:
        #
        #  xi >= Li (i in lowerB) : z[:nlowerB]
        #  xi <= Ui (i in upperB) : z[nlowerB:nlowerB+nupperB]
        #  xi >= Li (i in rangeB) : z[nlowerB+nupperB:nlowerB+nupperB+nrangeB]
        #  xi <= Ui (i in rangeB) : z[nlowerB+nupper+nrangeB:]

        # Set appropriate primal-dual starting point
        (self.x, self.z) = (nlp.x0, np.ones(self.nz))
        assert np.all(self.z > 0)


    def AtOptimality(self, x, z, **kwargs):
        """
        Shortcut for problems with bound constraints only.
        """
        return self.nlp.AtOptimality(x, np.array([]), z, **kwargs)


    def obj(self, x, z, **kwargs):
        """
        Evaluate the primal-dual merit function at (x,z):

        f(x) - mu * sum log(xi) + x'z - mu * sum (log(xi*zi/mu) + 1).

        If there are b >= 0
        one-sided bound constraints and q >= 0 two-sided bound constraints, the
        vector z should have length b+2q. The
        components z[i] (0 <= i < b+q) correspond, in increasing order of
        variable indices to
            - variables with a lower bound only,
            - variables with an upper bound only,
            - the lower bound on variables that are bounded below and above.

        The components z[i] (b+q <= i < b+2q) correspond to the upper bounds on
        variables that are bounded below and above.

        This function returns the value of the primal-dual merit function. The
        current value of the objective function can be supplied via the keyword
        argument `f`.
        """
        mu = kwargs.get('mu', self.mu)
        f = kwargs.get('f', self.nlp.obj(x))
        nlp = self.nlp ; n = nlp.n
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar
        lB = nlp.lowerB ; uB = nlp.upperB ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB

        merit = f

        # Include contribution from bound constraints.
        slB = x[lB] - Lvar[lB] ; zlB = z[:nlB]
        merit -= mu * np.sum(np.log(slB))
        merit += np.dot(slB, zlB)
        merit -= mu * np.sum(1 + np.log(slB * zlB/mu))

        suB = Uvar[uB] - x[uB] ; zuB = z[nlB:nlB+nuB]
        merit -= mu * np.sum(np.log(suB))
        merit += np.dot(suB, zuB)
        merit -= mu * np.sum(1 + np.log(suB * zuB/mu))

        srlB = x[rB] - Lvar[rB] ; zrlB = z[nlB+nuB:nlB+nuB+nrB]
        merit -= mu * np.sum(np.log(srlB))
        merit += np.dot(srlB, zrlB)
        merit -= mu * np.sum(1 + np.log(srlB * zrlB/mu))

        sruB = Uvar[rB] - x[rB] ; zruB = z[nlB+nuB+nrB:]
        merit -= mu * np.sum(np.log(sruB))
        merit += np.dot(sruB, zruB)
        merit -= mu * np.sum(1 + np.log(sruB * zruB/mu))

        return merit


    def grad(self, x, z, **kwargs):
        """
        Evaluate the gradient of the primal-dual merit function at (x,z).
        The gradient of the objective function at `x`, if known, can be passed
        via the keyword argument `g`.
        See :meth:`PDMerit` for a description of `z`.
        """
        mu = kwargs.get('mu', self.mu)
        check_optimal = kwargs.get('check_optimal', False)
        gf = kwargs.get('g', None)

        nlp = self.nlp ; nx = nlp.n ; nz = self.nz
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar
        lB = nlp.lowerB ; uB = nlp.upperB ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB

        g = np.empty(nx + nz)

        # Gradient of the objective function.
        if gf is None:
            g[:nx] = self.nlp.grad(x)
        else:
            g[:nx] = gf.copy()

        # Check optimality conditions at this point if requested.
        if check_optimal:
            res, self.optimal = self.AtOptimality(x, z, g=g[:nx])
            self.dRes = res[0] ; self.cRes = res[2] ; self.pRes = res[4]

        # Segment s and z for conciseness.
        slB  = x[lB] - Lvar[lB] ; zlB  = z[:nlB]
        suB  = Uvar[uB] - x[uB] ; zuB  = z[nlB:nlB+nuB]
        srlB = x[rB] - Lvar[rB] ; zrlB = z[nlB+nuB:nlB+nuB+nrB]
        sruB = Uvar[rB] - x[rB] ; zruB = z[nlB+nuB+nrB:]

        # Assemble the gradient with respect to x.
        g[lB] += -2 * mu / slB  + zlB
        g[uB] +=  2 * mu / suB  - zuB
        g[rB] += -2 * mu / srlB + zrlB
        g[rB] +=  2 * mu / sruB - zruB

        # Assemble the gradient with respect to z.
        n1 = nlB ; n2 = n1 + nuB ; n3 = n2 + nrB
        g[nx:nx+n1]    = slB  - mu / zlB
        g[nx+n1:nx+n2] = suB  - mu / zuB
        g[nx+n2:nx+n3] = srlB - mu / zrlB
        g[nx+n3:]      = sruB - mu / zruB

        return g


    def primal_hess(self, x, z, **kwargs):
        """
        Assemble the "primal" Hessian matrix of the merit function at (x,z):

        [ H + 2 mu X^{-2}      I     ]
        [      I           mu Z^{-2} ].
        """
        mu = kwargs.get('mu', self.mu)

        nlp = self.nlp ; nx = nlp.n ; nz = self.nz
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar
        lB = nlp.lowerB ; uB = nlp.upperB ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        rlB = self.rlB ; ruB = self.ruB ; rrB = self.rrB

        # Segment s and z for conciseness.
        slB  = x[lB] - Lvar[lB] ; zlB  = z[:nlB]
        suB  = Uvar[uB] - x[uB] ; zuB  = z[nlB:nlB+nuB]
        srlB = x[rB] - Lvar[rB] ; zrlB = z[nlB+nuB:nlB+nuB+nrB]
        sruB = Uvar[rB] - x[rB] ; zruB = z[nlB+nuB+nrB:]

        H = sp(nrow=nx+nz, ncol=nx+nz,
               sizeHint=self.nlp.nnzh+3*nz, symmetric=True)

        # Leading block: H + 2 * mu X^{-2}.
        H[:nx,:nx] = self.nlp.hess(x, self.nlp.pi0)
        H.addAt(2 * mu/slB**2,  lB, lB)
        H.addAt(2 * mu/suB**2,  uB, uB)
        H.addAt(2 * mu/srlB**2, rB, rB)
        H.addAt(2 * mu/sruB**2, rB, rB)

        # Bottom left block: identity-ish.
        n1 = nx + rlB       ; H.put( 1, n1, lB)
        n2 = nx + ruB       ; H.put(-1, n2, uB)
        n3 = nx + rrB       ; H.put( 1, n3, rB)
        n4 = nx + nrB + rrB ; H.put(-1, n4, rB)

        # Bottom right block: mu * Z^{-2}.
        H.put(mu/zlB**2,  n1, n1)
        H.put(mu/zuB**2,  n2, n2)
        H.put(mu/zrlB**2, n3, n3)
        H.put(mu/zruB**2, n4, n4)

        return H


    def primal_hprod(self, x, z, p, **kwargs):
        """
        Compute the matrix-vector product between the Hessian matrix of the
        primal-dual merit function at (x,z) and the vector p. See
        help(PDMerit) for a description of z. If there are b bounded variables
        and q two-sided bounds, the vector p should have length n+b+2q. The
        Hessian-vector product has the general form

        [ H + 2 mu X^{-2}      I     ] [ p1 ]   [ (H + 2 mu X^{-2}) p1 + p2 ]
        [      I           mu Z^{-2} ] [ p2 ] = [     p1 + mu Z^{-2} p2     ].
        """
        mu = kwargs.get('mu', self.mu)

        nlp = self.nlp ; nx = nlp.n ; nz = self.nz
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar
        lB = nlp.lowerB   ; uB = nlp.upperB   ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        rlB = self.rlB ; ruB = self.ruB ; rrB = self.rrB
        rrB2 = nlB + nuB + nrB + rrB

        # Segment s and z for conciseness.
        slB  = x[lB] - Lvar[lB] ; zlB  = z[:nlB]
        suB  = Uvar[uB] - x[uB] ; zuB  = z[nlB:nlB+nuB]
        srlB = x[rB] - Lvar[rB] ; zrlB = z[nlB+nuB:nlB+nuB+nrB]
        sruB = Uvar[rB] - x[rB] ; zruB = z[nlB+nuB+nrB:]

        Hp = np.empty(nx + nz) ; p1 = p[:nx] ; p2 = p[nx:]

        # Contributions from first block row.
        Hp[:nx] = self.nlp.hprod(x, self.nlp.pi0, p1)
        Hp[lB] += 2 * mu/slB**2 * p1[lB] + p2[rlB]
        Hp[uB] += 2 * mu/suB**2 * p1[uB] - p2[nlB+ruB]
        Hp[rB] += 2 * (mu/srlB**2 + mu/sruB**2) * p1[rB] + p2[rrB] - p2[rrB2]

        # Contributsions from second block row.
        Hp[nx+rlB]             =  p1[lB] + mu/zlB**2  * p2[rlB]
        Hp[nx+nlB+ruB]         = -p1[uB] + mu/zuB**2  * p2[ruB]
        Hp[nx+nlB+nuB+rrB]     =  p1[rB] + mu/zrlB**2 * p2[rrB]
        Hp[nx+nlB+nuB+nrB+rrB] = -p1[rB] + mu/zruB**2 * p2[rrB2]

        return Hp


    def primal_dual_hess(self, x, z, **kwargs):
        """
        Assemble the "primal-dual" Hessian matrix of the merit function
        at (x,z):

        [ H + 2 X^{-1} Z     I     ]
        [      I          Z^{-1} X ].
        """
        mu = kwargs.get('mu', self.mu)

        nlp = self.nlp ; nx = nlp.n ; nz = self.nz
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar
        lB = nlp.lowerB ; uB = nlp.upperB ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        rlB = self.rlB ; ruB = self.ruB ; rrB = self.rrB

        # Segment s and z for conciseness.
        slB  = x[lB] - Lvar[lB] ; zlB  = z[:nlB]
        suB  = Uvar[uB] - x[uB] ; zuB  = z[nlB:nlB+nuB]
        srlB = x[rB] - Lvar[rB] ; zrlB = z[nlB+nuB:nlB+nuB+nrB]
        sruB = Uvar[rB] - x[rB] ; zruB = z[nlB+nuB+nrB:]

        H = sp(nrow=nx+nz, ncol=nx+nz,
               sizeHint=self.nlp.nnzh+3*nz, symmetric=True)

        # Leading block: H + 2 * mu X^{-2}.
        H[:nx,:nx] = self.nlp.hess(x, self.nlp.pi0)
        H.addAt(2 * zlB / slB, lB, lB)
        H.addAt(2 * zuB / suB, uB, uB)
        H.addAt(2 * zrlB/srlB, rB, rB)
        H.addAt(2 * zruB/sruB, rB, rB)

        # Bottom left block: identity-ish.
        n1 = nx + rlB       ; H.put( 1, n1, lB)
        n2 = nx + ruB       ; H.put(-1, n2, uB)
        n3 = nx + rrB       ; H.put( 1, n3, rB)
        n4 = nx + nrB + rrB ; H.put(-1, n4, rB)

        # Bottom right block: mu * Z^{-2}.
        H.put(slB / zlB, n1, n1)
        H.put(suB / zuB, n2, n2)
        H.put(srlB/zrlB, n3, n3)
        H.put(sruB/zruB, n4, n4)

        return H


    def primal_dual_hprod(self, x, z, p, **kwargs):
        """
        Compute the matrix-vector product between the modified Hessian matrix of
        the primal-dual merit function at (x,z) and the vector p. See
        help(PDMerit) for a description of z. If there are b bounded variables
        and q two-sided bounds, the vector p should have length n+b+2q.
        The Hessian matrix has the general form

        [ H + 2 X^{-1} Z      I     ] [ p1 ]   [ (H + 2 X^{-1} Z) * p1 + p2 ]
        [      I           Z^{-1} X ] [ p2 ] = [     p1 + Z^{-1} X p3       ].
        """
        mu = kwargs.get('mu', self.mu)

        n = self.nlp.n
        ndual = self.ndual
        N = self.nlowerB + self.nupperB + self.nrangeB
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        Hp = np.zeros(n + self.ndual, 'd')

        Hp[:n] = self.nlp.hprod(x, self.nlp.pi0, p[:n])
        for i in self.lowerB:
            k = self.lowerB.index(i)
            Hp[i] += 2.0 * z[k]/(x[i]-Lvar[i]) * p[i] + p[n+k]
            Hp[n+k] += p[i] + (x[i]-Lvar[i])/z[k] * p[n+k]
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            Hp[i] += 2.0 * z[k]/(Uvar[i]-x[i]) * p[i] - p[n+k]
            Hp[n+k] += -p[i] - (Uvar[i]-x[i])/z[k] * p[n+k]
        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            Hp[i] += 2.0 * z[k]/(x[i]-Lvar[i]) * p[i] + p[n+k]
            Hp[n+k] += p[i] + (x[i]-Lvar[i])/z[k] * p[n+k]
            k += self.nrangeB
            Hp[i] += 2.0 * z[k]/(Uvar[i]-x[i]) * p[i] - p[n+k]
            Hp[n+k] += -p[i] + (Uvar[i]-x[i])/z[k] * p[n+k]

        nlp = self.nlp ; nx = nlp.n ; nz = self.nz
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar
        lB = nlp.lowerB   ; uB = nlp.upperB   ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        rlB = self.rlB ; ruB = self.ruB ; rrB = self.rrB
        rrB2 = nlB + nuB + nrB + rrB

        # Segment s and z for conciseness.
        slB  = x[lB] - Lvar[lB] ; zlB  = z[:nlB]
        suB  = Uvar[uB] - x[uB] ; zuB  = z[nlB:nlB+nuB]
        srlB = x[rB] - Lvar[rB] ; zrlB = z[nlB+nuB:nlB+nuB+nrB]
        sruB = Uvar[rB] - x[rB] ; zruB = z[nlB+nuB+nrB:]

        Hp = np.empty(nx + nz) ; p1 = p[:nx] ; p2 = p[nx:]

        # Contributions from first block row.
        Hp[:nx] = self.nlp.hprod(self.nlp.pi0, p1)
        Hp[lB] += 2 * zlB/slB * p1[lB] + p2[rlB]
        Hp[uB] += 2 * zuB/suB * p1[uB] - p2[nlB+ruB]
        Hp[rB] += 2 * (zrlB/srlB + zruB/sruB) * p1[rB] + p2[rrB] - p2[rrB2]

        # Contributsions from second block row.
        Hp[nx+rlB]             =  p1[lB] + slB/zlB   * p2[rlB]
        Hp[nx+nlB+ruB]         = -p1[uB] + suB/zuB   * p2[ruB]
        Hp[nx+nlB+nuB+rrB]     =  p1[rB] + srlB/zrlB * p2[rrB]
        Hp[nx+nlB+nuB+nrB+rrB] = -p1[rB] + sruB/zruB * p2[rrB2]

        return Hp


    def _hess_template(self, **kwargs):
        """
        Assemble the part of the modified Hessian matrix of the primal-dual
        merit function that is iteration independent.
        """
        nlp = self.nlp ; nx = nlp.n ; nz = self.nz
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar
        lB = nlp.lowerB ; uB = nlp.upperB ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; rlB = self.rlB
        nuB = nlp.nupperB ; ruB = self.ruB
        nrB = nlp.nrangeB ; rrB = self.rrB

        B = sp(nrow=nx+nz, ncol=nx+nz,
               sizeHint=self.nlp.nnzh+3*nz, symmetric=True)

        # Bottom left block: identity-ish.
        n1 = nx + rlB       ; B.put( 1, n1, lB)
        n2 = nx + ruB       ; B.put(-1, n2, uB)
        n3 = nx + rrB       ; B.put( 1, n3, rB)
        n4 = nx + nrB + rrB ; B.put(-1, n4, rB)

        return B



class PrimalDualInteriorPointFramework:

    def __init__(self, merit, TR, TrSolver, **kwargs):
        """
        Solve the bound-constrained problem

          minimize f(x)  subject to  l <= x <= u.

        The method is based on the primal-dual merit function of
        Forsgren and Gill (1998). For now, only bound-constrained problems are
        supported.
        """

        self.merit = merit
        self.TR = TR
        self.TrSolver = TrSolver

        self.bound_rel_factor = 0.1
        self.bound_abs_factor = 0.1

        self.verbose       = kwargs.get('verbose',  False)
        self.explicit      = kwargs.get('explicit', False)  # Form Hessian or not.
        self.maxiter       = kwargs.get('maxiter', max(100,2*self.merit.nlp.n))
        self.ny            = kwargs.get('ny',      False)
        self.inexact       = kwargs.get('inexact', False)
        self.nyMax         = kwargs.get('nyMax',   5)
        self.opportunistic = kwargs.get('opportunistic', True)
        self.muerrfact     = kwargs.get('muerrfact', 10)
        self.mu_min = 1.0e-09

        # Assemble the part of the primal-dual Hessian matrix that is constant.
        self.B = None
        if explicit:
            self.B = merit.hess_template()

        self.iter   = 0
        self.cgiter = 0
        self.f      = None   # Objective function value.
        self.psi    = None   # Merit function value.
        self.gf     = None   # Objective function gradient.
        self.g      = None   # Merit function gradient.
        self.g_old  = None   # A previous gradient we may wish to keep around.
        self.gNorm  = None   # Merit function gradient norm.
        self.save_g = False

        self.hformat = '%-5s %8s %7s %7s %7s %7s %5s %8s %8s %4s\n'
        head = ('Iter','f(x)','Resid','gPhi','mu','alpha',
                'cg','rho','Delta','Stat')
        self.header  = self.hformat % head
        self.hlen    = len(self.header)
        self.hline   = '-' * self.hlen + '\n'
        self.itFormat = '%-5d  '
        self.format='%8.1e %7.1e %7.1e %7.1e %7.1e %5d %8.1e %8.1e %4s\n'
        self.printFrequency = 20

        # Optimality residuals, updated along the iterations
        self.kktRes = None
        self.optimal = False
        return


    def get_feasible_point(self, **kwargs):
        """
        Compute a strictly feasible initial primal-dual estimate (x,z).
        By default, x is taken as the starting point given in `nlp` and moved
        strictly inside the bounds, and z is taken as the vector of ones.
        """
        nlp = self.merit.nlp
        n = nlp.n
        Lvar = nlp.Lvar ; lB = nlp.lowerB
        Uvar = nlp.Uvar ; uB = nlp.upperB ; rB = nlp.rangeB
        bnd_rel = self.bound_rel_factor ; bnd_abs = self.bound_abs_factor

        x = nlp.x0[:]

        # Compute primal starting guess.
        x[lB] = np.maximum(x[lB],
                           (1+np.sign(Lvar[lB]) * bnd_rel)*Lvar[lB] + bnd_abs)
        x[uB] = np.maximum(x[uB],
                           (1-np.sign(Uvar[uB]) * bnd_rel)*Uvar[uB] + bnd_abs)
        x[rB] = (Lvar[rB] + Uvar[rB])/2

        # Compute dual starting guess.
        z = np.ones(self.merit.nz)

        return (x,z)


    def primal_multipliers(self, x, **kwargs):
        """
        Return the vector of primal multipliers at `x`. The value of the
        barrier parameter used can either be supplied using the keyword
        argument `mu` or the current value of the instance is used.
        """
        mu = kwargs.get('mu', self.mu)
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lB = self.lowerB ; uB = self.upperB ; rB = self.rangeB
        nlB = self.nlowerB ; nuB = self.nupperB ; nrB = self.nrangeB

        z = np.empty(self.nz)
        z[:nlB] = mu/(x[lB]-Lvar[lB])
        z[nlB:nlB+nuB] = mu/(Uvar[uB]-x[uB])
        z[nlB+nuB:nlB+nuB+nrB] = mu/(x[rB]-Lvar[rB])
        z[nlB+nuB+nrB:] = mu/(Uvar[rB]-x[rB])

        return z


    def ftb(self, x, z, step, **kwargs):
        """
        Compute the largest alpha in ]0,1] such that::

            (x,z) + alpha * step >= (1 - tau) * (x,z)

        where 0 < tau < 1. By default, tau = 0.9.
        """
        tau = kwargs.get('tau', 0.9)
        n = self.nlp.n
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lowerB = self.lowerB ; upperB = self.upperB ; rangeB = self.rangeB

        #pdb.set_trace()

        dx = step[:n]  # Step in x.
        lowerneg = np.where(dx[lowerB] < 0)[0]
        upperneg = np.where(dx[upperB] < 0)[0]
        rangeneg = np.where(dx[rangeB] < 0)[0]
        stpmax = 1.0
        if len(lowerneg) > 0:
            idx = lowerB[lowerneg]
            stpmax = min(stpmax, np.min(-tau*(x[idx]-Lvar[idx])/dx[idx]))
        if len(upperneg) > 0:
            idx = upperB[upperneg]
            stpmax = min(stpmax, np.min(-tau*(Uvar[idx]-x[idx])/dx[idx]))
        if len(rangeneg) > 0:
            idx = rangeB[rangeneg]
            stpmax = min(stpmax, np.min(-tau*(x[idx]-Lvar[idx])/dx[idx]))
            stpmax = min(stpmax, np.min(-tau*(Uvar[idx]-x[idx])/dx[idx]))

        stpx = stpmax

        dz = step[n:]  # Step in z.
        stpmax = 1.0
        whereneg = np.where(dz < 0)[0]
        if len(whereneg) > 0:
            stpmax = min(stpmax, np.min(-tau * z[whereneg]/dz[whereneg]))

        return stpx, stpmax


    def SetupPrecon(self, **kwargs):
        """
        Construct or set up the preconditioner---must be overridden.
        """
        return None


    def Precon(self, v, **kwargs):
        """
        Generic preconditioning method---must be overridden.
        """
        return v


    def UpdatePrecon(self, **kwargs):
        """
        Override this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None


    def SolveInner(self, **kwargs):
        """
        Perform a series of inner iterations so as to minimize the primal-dual
        merit function with the current value of the barrier parameter to within
        some given tolerance. The only optional argument recognized is

            stopTol     stopping tolerance (default: muerrfact * mu).
        """

        merit = self.merit ; nx = merit.nlp.n ; nz = merit.nz
        rho = 1                  # Dummy initial value for rho
        niter = 0                # Dummy initial value for number of inners
        status = ''              # Dummy initial step status
        alpha = 0.0              # Fraction-to-the-boundary step size
        if self.inexact:
            cgtol = 1.0
        else:
            cgtol = -1.0
        inner_iter = 0           # Inner iteration counter

        # Obtain starting point.
        (x,z) = (self.x, self.z)

        # Obtain first-order data at starting point.
        if self.iter == 0:
            f = nlp.obj(x) ; gf = nlp.grad(x)
        else:
            f = self.f ; gf = self.gf

        psi = merit.obj(x, z, f=f)
        g = merit.grad(x, z, g=gf, check_optimal=True)
        gNorm = norm2(g)
        if self.optimal: return

        # Reset initial trust-region radius
        self.TR.Delta = 0.1*gNorm

        # Set inner iteration stopping tolerance
        stopTol = kwargs.get('stopTol', self.muerrfact * self.mu)
        finished = (gNorm <= stopTol) or (self.iter >= self.maxiter)

        while not finished:

            # Print out header every so many iterations
            if self.verbose:
                if self.iter % self.printFrequency == 0:
                    sys.stdout.write(self.hline)
                    sys.stdout.write(self.header)
                    sys.stdout.write(self.hline)

                if inner_iter == 0:
                    sys.stdout.write(('*' + self.itFormat) % self.iter)
                else:
                    sys.stdout.write((' ' + self.itFormat) % self.iter)

                sys.stdout.write(self.format % (f,
                                 max(norm_infty(self.dRes),
                                     norm_infty(self.cRes),
                                     norm_infty(self.pRes)),
                                 gNorm,
                                 self.mu, alpha, niter, rho,
                                 self.TR.Delta, status))

            # Set stopping tolerance for trust-region subproblem.
            if self.inexact:
                cgtol = max(1.0e-8, min(0.1 * cgtol, sqrt(gNorm)))
                if self.debug: self._debugMsg('cgtol = ' + str(cgtol))

            # Update Hessian matrix with current iteration information.
            # self.PDHess(x,z)

            if self.debug:
                self._debugMsg('g = ' + np.str(g))
                self._debugMsg('gNorm = ' + str(gNorm))
                self._debugMsg('stopTol = ' + str(stopTol))
                self._debugMsg('dRes = ' + np.str(self.dRes))
                self._debugMsg('cRes = ' + np.str(self.cRes))
                self._debugMsg('pRes = ' + np.str(self.pRes))
                self._debugMsg('optimal = ' + str(self.optimal))

            # Set up the preconditioner if applicable.
            self.SetupPrecon()

            # Iteratively minimize the quadratic model in the trust region
            # m(s) = <g, s> + 1/2 <s, Hs>
            # Note that m(s) does not include f(x): m(0) = 0.
            H = SimpleLinearOperator(nx+nz, nx+nz, lambda v: merit.hprod(v),
                                     symmetric=True)
            subsolver = self.TrSolver(g, H,
                                   prec = self.Precon,
                                   radius = self.TR.Delta,
                                   reltol = cgtol,
                                   #fraction = 0.5,
                                   itmax = 2*(n+nz),
                                   #debug=True,
                                   #btol=.9,
                                   #cur_iter=np.concatenate((x,z))
                                   )
            subsolver.Solve()

            if self.debug:
                self._debugMsg('x = ' + np.str(x))
                self._debugMsg('z = ' + np.str(z))
                self._debugMsg('step = ' + np.str(solver.step))
                self._debugMsg('step norm = ' + str(solver.stepNorm))

            # Record total number of CG iterations.
            niter = subsolver.niter
            self.cgiter += subsolver.niter

            # Compute maximal step to the boundary and next candidate.
            alphax, alphaz = self.ftb(x, z, solver.step)
            alpha = min(alphax, alphaz)
            dx = subsolver.step[:n] ; dz = subsolver.step[n:]
            x_trial = x + alphax * dx
            z_trial = z + alphaz * dz
            f_trial = merit.nlp.obj(x_trial)
            psi_trial = merit.obj(x_trial, z_trial, f=f_trial)

            # Compute ratio of predicted versus achieved reduction.
            rho  = self.TR.Rho(psi, psi_trial, subsolver.m)

            if self.debug:
                self._debugMsg('m = ' + str(solver.m))
                self._debugMsg('x_trial = ' + np.str(x_trial))
                self._debugMsg('z_trial = ' + np.str(z_trial))
                self._debugMsg('psi_trial = ' + str(psi_trial))
                self._debugMsg('rho = ' + str(rho))

            # Accept or reject next candidate
            status = 'Rej'
            if rho >= self.TR.eta1:
                self.TR.UpdateRadius(rho, solver.stepNorm)
                x = x_trial
                z = z_trial
                f = f_trial
                psi = psi_trial
                gf = nlp.grad(x)
                g = self.GradPDMerit(x, z, g=gf, check_optimal=True)
                gNorm = norm2(g)
                if self.optimal:
                    finished = True
                    continue
                status = 'Acc'
            else:
                if self.ny: # Backtracking linesearch a la "Nocedal & Yuan"
                    slope = np.dot(g, solver.step)
                    target = psi + 1.0e-4 * alpha * slope
                    j = 0

                    while (psi_trial >= target) and (j < self.nyMax):
                        alphax /= 1.2 ; alphaz /= 1.2
                        alpha = min(alphax, alphaz)
                        target = psi + 1.0e-4 * alpha * slope
                        x_trial = x + alphax * dx
                        z_trial = z + alphaz * dz
                        f_trial = nlp.obj(x_trial)
                        psi_trial = self.PDMerit(x_trial, z_trial, f=f_trial)
                        j += 1

                    if self.opportunistic or (j < self.nyMax):
                        x = x_trial
                        z = z_trial #self.PrimalMultipliers(x)
                        f = f_trial
                        psi = psi_trial
                        gf = nlp.grad(x)
                        g = self.GradPDMerit(x, z, g=gf, check_optimal=True)
                        gNorm = norm2(g)
                        if self.optimal:
                            finished = True
                            continue
                        self.TR.Delta = alpha * solver.stepNorm
                        status = 'N-Y'

                    else:
                        self.TR.UpdateRadius(rho, solver.stepNorm)

                else:
                    self.TR.UpdateRadius(rho, solver.stepNorm)

            self.UpdatePrecon()
            self.iter += 1
            inner_iter += 1
            finished = (gNorm <= stopTol) or (self.iter >= self.maxiter)
            if self.debug: sys.stderr.write('\n')

        # Store final iterate
        (self.x, self.z) = (x, z)
        self.f = f
        self.gf = gf
        self.g = g
        self.gNorm = gNorm
        self.psi = psi
        return


    def SolveOuter(self, **kwargs):

        nlp = self.nlp
        n = nlp.n

        err = min(nlp.stop_d, nlp.stop_c, nlp.stop_p)
        self.mu_min = min(self.mu_min, (err/(1 + self.muerrfact))/2)

        # Measure solve time
        t = cputime()

        # Solve sequence of inner iterations
        while (not self.optimal) and (self.mu >= self.mu_min) and \
                (self.iter < self.maxiter):
            self.SolveInner() #stopTol=max(1.0e-7, 5*self.mu))
            #self.z = self.PrimalMultipliers(self.x)
            #res, self.optimal = self.AtOptimality(self.x, self.z)
            self.UpdateMu()

        self.tsolve = cputime() - t    # Solve time
        if self.optimal:
            print 'First-order optimal solution found'
        elif self.iter >= self.maxiter:
            print 'Maximum number of iterations reached'
        else:
            print 'Reached smallest allowed value of barrier parameter'
            #print '(mu = %8.2e, mu_min = %8.2e)' % (self.mu, self.mu_min)
        return


    def UpdateMu(self, **kwargs):
        """
        Update the barrier parameter before the next round of inner iterations.
        """
        self.mu /= 5
        #res = max(self.dRes, self.cRes)
        #guard = min(res/5.0, res**(1.5))
        #if guard <= self.mu/5.0:
        #    self.mu = guard
        #else:
        #    self.mu = min(self.mu/5.0, self.mu**(1.5))
        #self.mu = min(self.mu/5.0, self.mu**(1.5))
        #self.mu = min(self.mu/2.0, max(self.mu/5.0, res/5.0))
        return None


    def _debugMsg(self, msg):
        sys.stderr.write('Debug:: ' + msg + '\n')
        return None


# Wrapper around merit function class to pretend it takes a single argument.
# For derivative checker.
class _meritfunction(PrimalDualMeritFunction):

    def __init__(self, nlp, **kwargs):
        PrimalDualMeritFunction.__init__(self, nlp, **kwargs)
        # Number of variables of the merit function.
        # Not to be confused with nlp.n!
        # This is solely for the purpose of the derivative checker.
        self.n = nlp.n + self.nz
        self.m = 0

    def obj(self, xz, **kwargs):
        nx = self.nlp.n
        return PrimalDualMeritFunction.obj(self, xz[:nx], xz[nx:], **kwargs)

    def grad(self, xz, **kwargs):
        nx = self.nlp.n
        return PrimalDualMeritFunction.grad(self, xz[:nx], xz[nx:], **kwargs)

    def hess(self, xz, *args, **kwargs):
        nx = self.nlp.n
        return PrimalDualMeritFunction.primal_hess(self, xz[:nx], xz[nx:],
                                                   **kwargs)

    def hprod(self, xz, *args, **kwargs):
        nx = self.nlp.n
        return PrimalDualMeritFunction.primal_hprod(self, xz[:nx], xz[nx:],
                                                    p, **kwargs)


if __name__ == '__main__':

    from nlpy.model import AmplModel
    from nlpy.krylov.linop import PysparseLinearOperator
    from nlpy.optimize.tr.trustregion import TrustRegionFramework, TrustRegionCG
    from nlpy.tools.dercheck import DerivativeChecker

    # Set printing standards for arrays.
    np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    prob = sys.argv[1]

    # Initialize problem.
    nlp = AmplModel(prob)
    pdmerit = _meritfunction(nlp)
    (nx, nz) = (pdmerit.nlp.n, pdmerit.nz)

    # Initialize trust region framework.
    TR = TrustRegionFramework(Delta = 1.0,
                              eta1 = 0.0001,
                              eta2 = 0.95,
                              gamma1 = 1.0/3,
                              gamma2 = 2.5)

    # Set up interior-point framework.
    TRIP = PrimalDualInteriorPointFramework(
                pdmerit,
                TR,
                TrustRegionCG,
                silent = False,
#                #ny = True,
#                #inexact = True,
#                debug = False,
#                maxiter = 150,
#                mu = 1.0
                )

    (x, z) = TRIP.get_feasible_point()
    xz0 = np.concatenate((x,z))
    print 'Initial  x: ', x
    print 'Initial mu: ', pdmerit.mu
    print 'Initial  z: ', z
    print 'merit(x,z)= ', pdmerit.obj(xz0)
    print 'grad(x,z) = ', pdmerit.grad(xz0)
    H = pdmerit.hess(xz0)
    print 'hess(x,z) = ' ; print H

    # Check derivatives at initial point.
    derchk = DerivativeChecker(pdmerit, xz0)
    derchk.check(verbose=True, hess=True, jac=False, chess=False)

    # Test Hessian-vector products.
    err_matvec = 0
    Hop = PysparseLinearOperator(H)
    for i in range(10):
        p = np.random.randn(nx+nz)
        hp1 = Hop*p
        hp2 = pdmerit.hprod(xz0, p)
        err = np.linalg.norm(hp1-hp2)/(1 + np.linalg.norm(hp1))
        if err > 1.0e-12:
            err_matvec += 1
            print 'Inaccurate matvec:'
            print hp1
            print hp2

    print 'Found %d inaccurate matvecs.' % err_matvec

    # Solve problem
    # TRIP.SolveOuter()

    # Display final statistics
    # print 'Final variables:'; print TRIP.x
    # print 'Final multipliers:'; print TRIP.z
    # print
    # print 'Optimal: ', TRIP.optimal
    # print 'Variables: ', TRIP.nlp.n
    # print '# lower, upper, 2-sided bounds: %-d, %-d, %-d' % \
            #    (TRIP.nlowerB, TRIP.nupperB, TRIP.nrangeB)
    # print 'Primal feasibility error  = %15.7e' % TRIP.pRes
    # print 'Dual   feasibility error  = %15.7e' % TRIP.dRes
    # print 'Complementarity error     = %15.7e' % TRIP.cRes
    # print 'Number of function evals  = %d' % TRIP.nlp.feval
    # print 'Number of gradient evals  = %d' % TRIP.nlp.geval
    # print 'Number of Hessian  evals  = %d' % TRIP.nlp.Heval
    # print 'Number of matvec products = %d' % TRIP.nlp.Hprod
    # print 'Final objective value     = %15.7e' % TRIP.f
    # print 'Solution time: ', TRIP.tsolve

    nlp.close()
