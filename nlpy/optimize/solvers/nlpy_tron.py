#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import AmplModel
from nlpy.optimize.solvers.tron import TronFramework
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import nlpy.tools.logs
import sys, logging, os

# Create root logger.
log = logging.getLogger('tron')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Configure the solver logger.
sublogger = logging.getLogger('tron.solver')
sublogger.setLevel(logging.INFO)
sublogger.addHandler(hndlr)
sublogger.propagate = False


def pass_to_tron(nlp, **kwargs):

    t = cputime()
    tron = TronFramework(nlp, logger_name='tron.solver', **kwargs)
    t_setup = cputime() - t    # Setup time.

    tron.Solve()

    return (t_setup, tron)


usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent unconstrained nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("--reltol", action="store", type="float",
                  default=1.0e-6, dest="reltol",
                  help="Relative stopping tolerance")
parser.add_option("--abstol", action="store", type="float",
                  default=1.0e-6, dest="abstol",
                  help="Absolute stopping tolerance")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxit",  help="Specify maximum number of iterations")

# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxit'] = options.maxit
opts['reltol'] = options.reltol
opts['abstol'] = options.abstol

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False

if multiple_problems:
    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %15s %7s %7s %6s %6s %5s'
    hdr = hdrfmt % ('Name','Iter','Feval','Objective','dResid','pResid',
                    'Setup','Solve','Opt')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %15.8e %7.1e %7.1e %6.2f %6.2f %5s'
    log.info(hdr)
    log.info('-' * lhdr)

# Solve each problem in turn.
for ProblemName in args:

    nlp = AmplModel(ProblemName)

    # Check for equality-constrained problem.
    if nlp.m > 0:
        msg = '%s has %d linear or nonlinear constraints\n'
        log.error(msg % (nlp.name, nlp.nbounds, n_ineq))
        error = True
    else:
        ProblemName = os.path.basename(ProblemName)
        if ProblemName[-3:] == '.nl':
            ProblemName = ProblemName[:-3]
        t_setup, tron = pass_to_tron(nlp, **opts)
    nlp.close()  # Close model.

if not multiple_problems and not error:
    # Output final statistics
    log.info('--------------------------------')
    log.info('Tron: End of Execution')
    log.info('  Problem                      : %-s' % ProblemName)
    log.info('  Number of variables          : %-d' % nlp.n)
    log.info('  Initial/Final Objective      : %-g/%-g' % (tron.f0, tron.f))
    log.info('  Number of iterations         : %-d' % tron.iter)
    log.info('  Number of function evals     : %-d' % tron.nlp.feval)
    log.info('  Number of gradient evals     : %-d' % tron.nlp.geval)
    #log.info('  Number of Hessian  evals     : %-d' % tron.nlp.Heval)
    log.info('  Number of Hessian matvecs    : %-d' % tron.nlp.Hprod)
    log.info('  Setup/Solve time             : %-gs/%-gs' % (t_setup, tron.tsolve))
    log.info('  Total time                   : %-gs' % (t_setup + tron.tsolve))
    log.info('--------------------------------')
