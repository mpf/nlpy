"""
 TRON
 M. P. Friedlander and D. Orban, Banff, June 2012
"""
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.tools import norms
from nlpy.tools.timing import cputime
from nlpy.tools.exceptions import UserExitRequest
from pytron import dtron
from math import sqrt
import numpy
import logging
import pdb

__docformat__ = 'restructuredtext'

class troninter:

    def __init__(self, n):
        self.n       = n
        self.xc      = numpy.empty(n)
        self.s       = numpy.empty(n)
        self.indfree = numpy.empty(n)
        self.isave   = numpy.empty(3, dtype=numpy.int32)
        self.dsave   = numpy.empty(3)
        self.wa      = numpy.empty(7*n)
        self.wx      = numpy.empty(n)
        self.wy      = numpy.empty(n)
        self.iwa     = numpy.empty(3*n, dtype=numpy.int32)
        self.cgiter  = 0

    def solve(self, task, x, xl, xu, f, g, aprod, delta,
              frtol=1.0e-12, fatol=0.0, fmin=-1.0e+32, cgtol=0.1, itermax=None):

        if itermax is None:
            itermax = self.n
            
        x, delta, task, self.xc, self.s, self.indfree, self.isave, self.dsave, \
            self.wa, self.wx, self.wy, self.iwa = \
            dtron(x, xl, xu, f, g, aprod, delta, task, self.xc, self.s,
                  self.indfree, self.isave, self.dsave, self.wa, self.wx,
                  self.wy, self.iwa, frtol, fatol, fmin, cgtol, itermax)

        task = task.strip()    
        predred = self.dsave[2]
        self.cgiter = self.isave[2] - self.cgiter # record the current no. of CG its
        return (task, x, f, g, predred, self.cgiter)
        
        
class TronFramework:
    """
    An abstract framework Tron. Instantiate using

    `TRON = TronFramework(nlp)`

    :parameters:

        :nlp:   a :class:`NLPModel` object representing the problem. For
                instance, nlp may arise from an AMPL model

    :keywords:

        :x0:           starting point                    (default nlp.x0)
        :reltol:       relative stopping tolerance       (default `nlp.stop_d`)
        :abstol:       absolute stopping tolerance       (default 1.0e-6)
        :maxit:        maximum number of iterations      (default max(1000,10n))
        :inexact:      use inexact Newton stopping tol   (default False)
        :logger_name:  name of a logger object that can be used in the post
                       iteration                         (default None)
        :verbose:      print log if True                 (default True)

    Once a `TronFramework` object has been instantiated and the problem is
    set up, solve problem by issuing a call to `TRON.solve()`. The algorithm
    stops as soon as the Euclidian norm of the gradient falls below

        ``max(abstol, reltol * g0)``

    where ``g0`` is the Euclidian norm of the projected gradient at the
    initial point.
    """

    def __init__(self, nlp, **kwargs):

        self.nlp    = nlp
        self.iter   = 0         # Iteration counter
        self.total_cgiter = 0
        self.x      = kwargs.get('x0', self.nlp.x0.copy())
        self.maxit  = kwargs.get('maxit', max(1000, 10*self.nlp.n))
        self.f      = None
        self.f0     = None
        self.g      = None
        self.gpnorm = None
        self.task   = None

        self.tron = troninter(nlp.n)

        self.reltol  = kwargs.get('reltol', self.nlp.stop_d)
        self.abstol  = kwargs.get('abstol', 1.0e-6)
        self.verbose = kwargs.get('verbose', True)
        self.logger  = kwargs.get('logger', None)

        self.format  = '%-5d  %9.2e  %7.1e  %5i  %8.1e  %8.1e'
        self.format0 = '%-5d  %9.2e  %7.1e  %5s  %8s  %8s'
        self.hformat = '%-5s  %9s  %7s  %5s  %8s  %8s'
        self.header  = self.hformat % ('Iter','f(x)','|Pg(x)|','cg','PredRed','Radius')
        self.hlen    = len(self.header)
        self.hline   = '-' * self.hlen

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.tron')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        if not self.verbose:
            self.log.propagate=False

    def _task_is(self, task):
        return self.task[0] == task[0]

    def _gpnorm2(self, x, g, Lvar, Uvar):
        """
        Compute 2-norm of the projected gradient.
        """
        ineq =  Lvar != Uvar
        lowr =  numpy.logical_and(x == Lvar, ineq)
        uppr =  numpy.logical_and(x == Uvar, ineq)
        free =  numpy.logical_and(numpy.logical_and(~lowr, ~uppr), ineq)
        gpnrm2  = norms.norm2( numpy.minimum( g[lowr], 0. ) )**2
        gpnrm2 += norms.norm2( numpy.maximum( g[uppr], 0. ) )**2
        gpnrm2 += norms.norm2(                g[free]       )**2
        return sqrt(gpnrm2)
        
    def hprod(self, v, **kwargs):
        """
        Default hprod based on nlp's hprod. User should overload to
        provide a custom routine, e.g., a quasi-Newton approximation.
        """
        return self.nlp.hprod(self.x, self.nlp.pi0, v)

    def precon(self, v, **kwargs):
        """
        Generic preconditioning method---must be overridden.
        Not yet implemented.
        """
        return v

    def PostIteration(self, **kwargs):
        """
        Override this method to perform work at the end of an iteration. For
        example, use this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def Solve(self, **kwargs):

        nlp = self.nlp

        # Project initial point into the box.
        self.x[self.x < nlp.Lvar] = nlp.Lvar[self.x < nlp.Lvar]
        self.x[self.x > nlp.Uvar] = nlp.Uvar[self.x > nlp.Uvar]
        
        # Gather initial information.
        self.f      = self.nlp.obj(self.x)
        self.f0     = self.f
        self.g      = self.nlp.grad(self.x)
        self.gpnorm = self._gpnorm2(self.x, self.g, nlp.Lvar, nlp.Uvar)

        stoptol = max(self.abstol, self.reltol * self.gpnorm)
        exitUser = False
        exitOptimal = self.gpnorm <= stoptol
        exitIter = self.iter >= self.maxit
        cgiter = 0
        
        t = cputime()

        # Print out header and initial log.
        if self.iter % 20 == 0 and self.verbose:
            self.log.info(self.hline)
            self.log.info(self.header)
            self.log.info(self.hline)
            self.log.info(self.format0 % (self.iter, self.f,
                                          self.gpnorm, '', '', ''))

        self.task = 'START'                 
        delta = norms.norm2(self.g)  # Initial trust-region radius.
        pdb.set_trace()
        while not (exitUser or exitOptimal or exitIter):

            print self.task

            self.task, self.x, self.f, self.g, predred, cgiter = \
                self.tron.solve(self.task, self.x, nlp.Lvar, nlp.Uvar, self.f, self.g,
                                self.hprod, delta, frtol=0,#self.reltol,
                                fatol=0,#self.abstol,
                                itermax=self.maxit)
            
            if self._task_is('F'):
                self.f = self.nlp.obj(self.x)
                
            elif self._task_is('G'):
                self.g = self.nlp.grad(self.x)
                self.gpnorm = self._gpnorm2(self.x, self.g, nlp.Lvar, nlp.Uvar)

            else:

                self.iter += 1
                exitIter = self.iter >= self.maxit
                exitOptimal = self.gpnorm <= stoptol
                
                try:
                    self.PostIteration()
                except UserExitRequest:
                    exitUser = True

                # Print out header, say, every 20 iterations
                if self.iter % 20 == 0 and self.verbose:
                    self.log.info(self.hline)
                    self.log.info(self.header)
                    self.log.info(self.hline)
                    
                if self.verbose:
                    self.log.info(self.format % (self.iter, self.f,
                                                 self.gpnorm, self.tron.isave[2],
                                                 predred, delta))                    
                    
        self.tsolve = cputime() - t    # Solve time 
        
        # Set final solver status.
        if exitUser:
            self.status = 'usr'
        elif exitOptimal:
            self.status = 'opt'
        else: # exitIter
            self.status = 'itr'

if __name__ == '__main__':
    import nlpy_tron
    