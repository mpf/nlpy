# -*- coding: utf-8 -*-
# Long-step primal-dual interior-point method for linear programming
# From Algorithm IPF on p.110 of Stephen J. Wright's book
# "Primal-Dual Interior-Point Methods", SIAM ed., 1997.
# The method uses the augmented system formulation. These systems
# are solved using PyMa27 or PyMa57.
#
# D. Orban, Montreal 2004. Revised September 2009.

from nlpy.model import SlackFramework
try:                            # To solve augmented systems
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext
from nlpy.tools import norms
from nlpy.tools import sparse_vector_class as sv
from nlpy.tools.timing import cputime

from pysparse import spmatrix
from pysparse.pysparseMatrix import PysparseMatrix
import numpy as np
import sys

class LPInteriorPointSolver:

    def __init__(self, lp, **kwargs):
        """
        Solve the linear program

           minimize     c'x
           subject to   Ax = b, x >= 0,

        where c is a sparse cost vector, A is a sparse constraint matrix and b is
        a dense right-hand side.

        The problem MUST be in standard form. No prior conversion is made.

        The values returned are

         x..............the final iterate
         y..............the final value of the Lagrange multipliers associated
                        to Ax=b
         s..............the final value of the Lagrange multipliers associated
                        to x>=0
         obj_value......the final cost
         iter...........the total number of iterations
         kktResid.......the final relative residual
         solve_time.....the time to solve the LP
         status.........a string describing the exit status.

         Keyword arguments may be passed to change options. They are as follows:

         debug                True/False (default: False)
              Will output some debugging info during the run.
              
         PredictorCorrector   True/False (default: True)
              Uses Mehrotra's predictor-corrector method. If set
              to False, a classical long-step method will be used.
              In both cases, the augmented system formulation is
              used and systems are solved with MA27.

         itermax              integer (default: 100)
              Sets the maximum number of iterations.

         tolerance            float (default: 1.0e-6)
              The algorithm terminates successfully as soon as the
              relative residual of the KKT conditions is smaller
              than 'tolerance' in infinity norm.          
        """
        self.lp = lp

        self.debug = kwargs.get('debug', False)

        self.A = lp.A()               # Constraint matrix
        #if not isinstance(self.A, PysparseMatrix):
        #    self.A = PysparseMatrix(matrix=self.A)

        m, n = self.A.shape

        # Residuals
        self.dFeas = np.zeros(n)
        self.pFeas = np.zeros(m)
        self.comp = np.zeros(n)

        self.b = lp.cons(self.dFeas)     # Right-hand side
        self.c = lp.cost()            # Cost vector
        self.c0 = 0 #lp.obj(self.dFeas)     # Constant term in objective
        self.normb  = norms.norm_infty(self.b)
        self.normc  = sv.norm_infty(self.c)
        self.normbc = 1 + max(self.normb, self.normc)

        # Initialize augmented matrix
        #self.H = PysparseMatrix(size=n+m, sizeHint=n+self.A.nnz, symmetric=True)
        self.H = spmatrix.ll_mat_sym(n+m, n+self.A.nnz)
        self.H[n:,:n] = self.A

        fmt_hdr = '%-4s  %9s  %-8s  %-7s  %-4s  %-4s  %-8s'
        self.header = fmt_hdr % ('Iter', 'Cost', 'Residual', 'Mu', 'AlPr', 'AlDu', 'LS Resid')
        self.format1 = '%-4d  %9.2e  %-8.2e'
        self.format2 = '  %-7.1e  %-4.2f  %-4.2f  %-8.2e'

        return

    def solve(self, x=None, **kwargs):

        lp = self.lp
        debug = self.debug
        itermax = kwargs.get('itermax', 10*lp.n)
        tolerance = kwargs.get('tolerance', 1.0e-6)
        PredictorCorrector = kwargs.get('PredictorCorrector', True)

        # Transfer pointers for convenience
        m, n = self.A.shape
        A = self.A ; b = self.b ; c = self.c ; H = self.H
        dFeas = self.dFeas ; pFeas = self.pFeas ; comp = self.comp

        # Initialize
        if x is None: x = self.set_initial_guess(self.lp, **kwargs)
        z = x.copy()
        y = np.zeros(m)
        dz = np.zeros(n)
        rhs = np.zeros(n+m)
        finished = False
        iter = 0

        sys.stdout.write(self.header + '\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        setup_time = cputime()

        # Main loop
        while not finished:

            if debug:
                sys.stderr.write(' z = ', z)
                sys.stderr.write(' x = ', x)

            # Compute residuals
            # pFeas = b - A*x
            A.matvec(x, pFeas)
            pFeas = b - pFeas
            #  comp = Xz
            comp = x*z
            #  dFeas = A^T y + z - c
            A.matvec_transp(y, dFeas)
            #dFeas = y * A  # This means  A^T y
            dFeas += z
            for k in c.keys(): dFeas[k] -= c[k]
    
            pResid = norms.norm_infty(pFeas)
            cResid = norms.norm_infty(comp)
            dResid = norms.norm_infty(dFeas)
            kktResid = max(pResid, cResid, dResid) / self.normbc
            obj_value = sv.dot(c, x)

            sys.stdout.write(self.format1 % (iter, obj_value, kktResid))
    
            if kktResid <= tolerance:
                status = 'Optimal solution found'
                finished = True
            elif  iter > itermax:
                status = 'Max number of iterations reached'
                finished = True
            else:
                # Solve the linear system
                #
                # [ -X^{-1} Z    A^T ]  [ dx ] = - [ dFeas - X^{-1} comp ]
                # [   A          0   ]  [ dy ]     [ -pFeas             ]
                #
                # and recover  dz = -X^{-1} (comp + Z dx)
                # with comp = Xs - sigma * mu e.

                # Compute augmented matrix and factorize it
                H.put(-z/x)  # In places (1,1), (2,2), ..., (n,n) by default
                LBL = LBLContext(H)

                # Compute mu
                mu = sum(comp)/n
                tau = max(.9995, 1.0-mu)

                if PredictorCorrector:
                    # Use Mehrotra predictor-corrector method
                    # Compute affine-scaling step, i.e. with sigma = 0
                    rhs[:n] = -(dFeas - z)
                    rhs[n:] = pFeas
                    (step, nres, neig) = self.solveSystem(LBL, rhs)
                    # Recover dx and dz
                    dx = step[:n]
                    dz = -z * (1 + dx/x)
                    # Compute primal and dual stepsizes
                    alpha_p = self.maxStepLength(x, dx)
                    alpha_d = self.maxStepLength(z, dz)
                    # Estimate duality gap after affine-scaling step
                    muAff = np.dot(x + alpha_p * dx, z + alpha_d * dz)/n
                    sigma = (muAff/mu)**3

                    if debug:
                        sys.stderr.write(' alpha_pAFF, alpha_dAFF, muAFF, sigma =', (alpha_p, alpha_d, muAff, sigma))

                    # Incorporate predictor information for corrector step
                    comp += dx*dz
                else:
                    # Use long-step method
                    # Compute right-hand side
                    sigma = min(0.1, 100*mu)

                # Assemble right-hand side
                comp -= sigma * mu
                dFeas -= comp/x
                rhs[:n] = -dFeas
                rhs[n:] = pFeas

                # Solve augmented system
                (step, nres, neig) = self.solveSystem(LBL, rhs)

                # Recover step
                dx = step[:n]
                dy = step[n:]
                dz = -(comp + z*dx)/x

                # Compute primal and dual stepsizes
                alpha_p = tau * self.maxStepLength(x, dx)
                alpha_d = tau * self.maxStepLength(z, dz)

                sys.stdout.write(self.format2 % (mu, alpha_p, alpha_d, nres))
                sys.stdout.write('\n')

                # Update iterates
                x += alpha_p * dx
                y += alpha_d * dy
                z += alpha_d * dz

                iter += 1

        solve_time = cputime() - setup_time
        sys.stdout.write('\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        self.x = x
        self.y = y
        self.z = z
        self.obj_value = obj_value + self.c0
        self.iter = iter
        self.kktResid = kktResid
        self.solve_time = solve_time
        self.status = status
        return

    def set_initial_guess(self, lp, **kwargs):
        # Compute initial guess
        bigM = max(self.A.norm('inf'), self.normbc)
        x = 100 * bigM * np.ones(lp.n)
        return x

    def maxStepLength(self, x, d):
        """
        Returns the max step length from x to the boundary
        of the nonnegative orthant in the direction d
        alpha_max = min [ 1, min { -x[i]/d[i] | d[i] < 0 } ].
        Note that 0 < alpha_max <= 1.
        """
        whereneg = np.nonzero(np.where(d < 0, d, 0))[0]
        dxneg = [-x[i]/d[i] for i in whereneg]
        dxneg.append(1)
        return min(dxneg)

    def solveSystem(self, LBL, rhs, itref_threshold=1.0e-5, nitrefmax=5):
        LBL.solve(rhs)
        nr = norms.norm2(LBL.residual)
        # If residual not small, perform iterative refinement
        LBL.refine(rhs, tol=itref_threshold, nitref=nitrefmax)
        nr1 = norms.norm2(LBL.residual)
        return (LBL.x, nr1, LBL.neig)


class RegLPInteriorPointSolver(LPInteriorPointSolver):

    def __init__(self, lp, **kwargs):
        """
        Solve a linear program of the form

           minimize    c' x
           subject to  A1 x + A2 s = b,                                 (LP)
                       s >= 0,

        where the variables s are slack variables. Any linear program may be
        converted to the above form by instantiation of the `SlackFramework`
        class. The conversion to the slack formulation is mandatory in this
        implementation.

        [ ... more information ... ]
        """

        if not isinstance(lp, SlackFramework):
            msg = 'Input problem must be an instance of SlackFramework'
            raise ValueError, msg

        # Initialize
        LPInteriorPointSolver.__init__(self, lp, **kwargs)
        self.regpr = kwargs.get('regpr', 1.0) ; self.regpr_min = 1.0e-8
        self.regdu = kwargs.get('regdu', 1.0) ; self.regdu_min = 1.0e-8

        # Record number of slack variables in LP
        self.nSlacks  = lp.n - lp.original_n

        # Initialize format strings for display
        fmt_hdr  = '%-4s  %9s  %-8s  %-8s  %-8s  %-8s  %-8s  %-7s  %-4s  %-4s'
        fmt_hdr += '  %-8s  %-8s  %-8s'
        self.header = fmt_hdr % ('Iter', 'Cost', 'pResid', 'dResid', 'cResid',
                                 'qNorm', 'rNorm', 'Mu', 'AlPr', 'AlDu',
                                 'LS Resid', 'RegPr', 'RegDu')
        self.format1 = '%-4d  %9.2e  %-8.2e  %-8.2e  %-8.2e  %-8.2e  %-8.2e'
        self.format2 = '  %-7.1e  %-4.2f  %-4.2f  %-8.2e  %-8.2e  %-8.2e'

        return

    def solve(self, **kwargs):

        lp = self.lp
        itermax = kwargs.get('itermax', 10*lp.n)
        tolerance = kwargs.get('tolerance', 1.0e-6)
        PredictorCorrector = kwargs.get('PredictorCorrector', True)

        # Transfer pointers for convenience.
        m, n = self.A.shape ; on = lp.original_n
        A = self.A ; b = self.b ; c = self.c ; H = self.H
        dFeas = self.dFeas ; pFeas = self.pFeas ; comp = self.comp
        regpr = self.regpr ; regdu = self.regdu
        debug = self.debug

        # Obtain initial point from Mehrotra's heuristic.
        (x,y,z) = self.set_initial_guess(self.lp, **kwargs)

        # Slack variables are the trailing variables in x.
        s = x[on:] ; ns = self.nSlacks

        # Initialize steps in dual variables.
        dz = np.zeros(ns)

        # Initialize perturbation vectors (q=primal, r=dual).
        q = np.zeros(n) ; qNorm = 0.0 ; rho_q = 0.0
        r = np.zeros(m) ; rNorm = 0.0 ; del_r = 0.0

        # Allocate room for right-hand side of linear systems.
        rhs = np.zeros(n+m)
        finished = False
        iter = 0

        # Display initial header.
        sys.stdout.write(self.header + '\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        setup_time = cputime()

        # Main loop.
        while not finished:

            if debug:
                sys.stderr.write(' z = ', z)
                sys.stderr.write(' x = ', x)

            # Compute residuals.
            A.matvec(x,pFeas) ; pFeas -= b ; pFeas *= -1  # pFeas = b - A x
            comp = s*z                                    # comp  = S z
            A.matvec_transp(y,dFeas) ; dFeas[on:] += z
            for k in c.keys(): dFeas[k] -= c[k]           # dFeas = A' y + z - c
    
            pResid = norms.norm_infty(pFeas)
            cResid = norms.norm_infty(comp)
            dResid = norms.norm_infty(dFeas)
            kktResid = max(pResid, cResid, dResid) / self.normbc
            obj_val = sv.dot(c, x)

            # Display objective and residual data.
            sys.stdout.write(self.format1 % (iter, obj_val, pResid, dResid,
                                             cResid, qNorm, rNorm))
    
            if kktResid <= tolerance:
                status = 'Optimal solution found'
                finished = True
                continue

            if iter > itermax:
                status = 'Maximum number of iterations reached'
                finished = True
                continue

            # Solve the linear system
            #
            # [ -pI          0          A1' ] [∆x] = [ c - A1' y              ]
            # [  0   -(S^{-1} Z + pI)   A2' ] [∆s]   [   - A2' y - µ S^{-1} e ]
            # [  A1          A2         dI  ] [∆y]   [ b - A1 x - A2 s        ]
            #
            # where s are the slack variables, p is the primal regularization
            # parameter, d is the dual regularization parameter, and
            # A = [ A1  A2 ]  where the columns of A1 correspond to the original
            # problem variables and those of A2 correspond to slack variables.
            #
            # We recover ∆z = -z - S^{-1} (Z ∆s + µ e).

            # Compute augmented matrix and factorize it.
            H.put(-regpr,       range(on))
            H.put(-z/s - regpr, range(on,n))
            H.put(regdu,        range(n,n+m))
            LBL = LBLContext(H, sqd=True)

            # Compute duality measure.
            mu = sum(comp)/ns
            tau = max(.9995, 1.0-mu)

            if PredictorCorrector:
                # Use Mehrotra predictor-corrector method.
                # Compute affine-scaling step, i.e. with centering = 0.
                rhs[:n]    = -dFeas
                rhs[on:n] += z
                rhs[n:]    =  pFeas
                (step, nres, neig) = self.solveSystem(LBL, rhs)
                
                # Recover dx and dz.
                dx = step[:n]
                ds = dx[on:]
                dz = -z * (1 + ds/s)

                # Compute largest allowed primal and dual stepsizes.
                alpha_p = self.maxStepLength(s, ds)
                alpha_d = self.maxStepLength(z, dz)

                # Estimate duality gap after affine-scaling step.
                muAff = np.dot(s + alpha_p * ds, z + alpha_d * dz)/ns
                sigma = (muAff/mu)**3

                if debug:
                    sys.stderr.write(' alpha_pAFF, alpha_dAFF, muAFF, sigma =',
                                     (alpha_p, alpha_d, muAff, sigma))

                # Incorporate predictor information for corrector step.
                comp += ds*dz
            else:
                # Use long-step method: Compute centering parameter.
                sigma = min(0.1, 100*mu)

            # Compute centering step with appropriate centering parameter.
            # Assemble right-hand side.
            comp -= sigma * mu
            dFeas[on:] -= comp/s
            rhs[:n] = -dFeas
            rhs[n:] =  pFeas

            # Solve augmented system.
            (step, nres, neig) = self.solveSystem(LBL, rhs)

            # Recover step.
            dx = step[:n]
            ds = dx[on:]
            dy = step[n:]
            dz = -z - (z*ds - sigma*mu)/s  # -(comp + z*dx)/x

            # Compute largest allowed primal and dual stepsizes.
            alpha_p = tau * self.maxStepLength(s, ds)
            alpha_d = tau * self.maxStepLength(z, dz)

            # Display data.
            sys.stdout.write(self.format2 % (mu, alpha_p, alpha_d,
                                             nres, regpr, regdu))
            sys.stdout.write('\n')

            # Update iterates.
            x += alpha_p * dx    # Also updates slack variables.
            y += alpha_d * dy
            z += alpha_d * dz
            q *= (1-alpha_p) ; q += alpha_p * dx
            r *= (1-alpha_d) ; r += alpha_d * dy
            qNorm = norms.norm_infty(q) ; rNorm = norms.norm_infty(r)

            # Update regularization parameters.
            regpr = max(regpr/2, self.regpr_min)
            regdu = max(regdu/2, self.regdu_min)

            iter += 1

        solve_time = cputime() - setup_time
        sys.stdout.write('\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        # Transfer final values to class members.
        self.x = x
        self.y = y
        self.z = z
        self.obj_value = obj_val + self.c0
        self.iter = iter
        self.kktResid = kktResid
        self.solve_time = solve_time
        self.status = status
        return

    def set_initial_guess(self, lp, **kwargs):
        """
        Compute initial guess according the Mehrotra's heuristic. Initial values
        of x are computed as the solution to the least-squares problem

          minimize ||s||  subject to  A1 x + A2 s = b

        which is also the solution to the augmented system

          [ 0   0   A1' ] [x]   [0]
          [ 0   I   A2' ] [s] = [0]
          [ A1  A2   0  ] [w]   [b].

        Initial values for (y,z) are chosen as the solution to the least-squares
        problem

          minimize ||z||  subject to  A1' y = c,  A2' y + z = 0

        which can be computed as the solution to the augmented system

          [ 0   0   A1' ] [w]   [c]
          [ 0   I   A2' ] [z] = [0]
          [ A1  A2   0  ] [y]   [0].

        To ensure stability and nonsingularity when A does not have full row
        rank, the (1,1) block is perturbed to 1.0e-4 * I and the (3,3) block is
        perturbed to -1.0e-4 * I.

        The values of s and z are subsequently adjusted to ensure they are
        positive. See [Methrotra, 1992] for details.
        """
        n = lp.n ; m = lp.m ; ns = self.nSlacks ; on = lp.original_n

        # Set up augmented system matrix and factorize it.
        self.H.put(1.0e-4, range(on))
        self.H.put(1.0, range(on,n))
        self.H.put(-1.0e-4, range(n,n+m))
        LBL = LBLContext(self.H, sqd=True)

        # Assemble first right-hand side and solve.
        rhs = np.zeros(n+m)
        rhs[n:] = self.b
        (step, nres, neig) = self.solveSystem(LBL, rhs)
        x = step[:n].copy()
        s = x[on:]  # Slack variables. Must be positive.

        # Assemble second right-hand side and solve.
        rhs[:] = 0.0
        for k in self.c.keys(): rhs[k] = self.c[k]
        (step, nres, neig) = self.solveSystem(LBL, rhs)
        y = step[n:].copy()
        z = step[on:n].copy()

        # Use Mehrotra's heuristic to ensure (s,z) > 0.
        if np.all(s >= 0):
            dp = 0.0
        else:
            dp = -1.5 * min(s[s < 0])
        if np.all(z >= 0):
            dd = 0.0
        else:
            dd = -1.5 * min(z[z < 0])

        if dp == 0.0: dp = 1.5
        if dd == 0.0: dd = 1.5

        es = sum(s+dp)
        ez = sum(z+dd)
        xs = sum((s+dp) * (z+dd))

        dp += 0.5 * xs/ez
        dd += 0.5 * xs/es
        s += dp
        z += dd

        if not np.all(s>0) or not np.all(z>0):
            raise ValueError, 'Initial point not strictly feasible'

        return (x,y,z)


############################################################

def usage():
    sys.stderr.write('Use: %-s problem_name\n' % sys.argv[0])
    sys.stderr.write(' where problem_name represents a linear program\n')


if __name__ == '__main__':

    from nlpy.model import AmplModel, SlackFramework

    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    probname = sys.argv[1]

    #lp = AmplModel(probname)
    lp = SlackFramework(probname)
    if not lp.islp():
        sys.stderr.write('Input problem must be a linear program\n')
        sys.exit(1)

    #lpSolver = LPInteriorPointSolver(lp)
    lpSolver = RegLPInteriorPointSolver(lp)
    lpSolver.solve(itermax=20, tolerance=1.0e-5)

    print 'Final x: ', lpSolver.x
    print 'Final y: ', lpSolver.y
    print 'Final z: ', lpSolver.z

    sys.stdout.write('\n' + lpSolver.status + '\n')
    sys.stdout.write(' #Iterations: %-d\n' % lpSolver.iter)
    sys.stdout.write(' RelResidual: %7.1e\n' % lpSolver.kktResid)
    sys.stdout.write(' Final cost : %7.1e\n' % lpSolver.obj_value)
    sys.stdout.write(' Solve time : %6.2fs\n' % lpSolver.solve_time)

    # End
    lp.close()