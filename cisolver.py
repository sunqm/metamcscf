import os
import commands
import tempfile
import numpy
import pyscf.tools.fcidump

cicallee = os.path.dirname(__file__) + '/callee.sh'

class CISolver(object):
    def __init__(self, parameters=None):
        self.parameters = parameters
        self._wfnfile = tempfile.NamedTemporaryFile()
        self.wfnfile = self._wfnfile.name

    def kernel(self, h1eff, eri_cas, ncas, nelecas, ci0=None):
        fcidumpfile = tempfile.NamedTemporaryFile()
        pyscf.tools.fcidump.from_integrals(fcidumpfile.name, h1eff, eri_cas,
                                           ncas, nelecas, nuc=0, ms=0)
        fcidumpfile.flush()
        cmd = [cicallee]
        cmd.append('--fcidump %s' % fcidumpfile.name)
        cmd.append('--wfnfile %s' % self.wfnfile)
        if ci0:
            cmd.append('--initguess %s' % ci0)
        e = float(commands.getoutput(' '.join(cmd)))
        return e, self.wfnfile

    def make_rdm12(self, wfnfile, ncas, nelecas):
        rdm1file = tempfile.NamedTemporaryFile()
        rdm2file = tempfile.NamedTemporaryFile()
        cmd = [cicallee]
        cmd.append('--rdm1file %s' % rdm1file.name)
        cmd.append('--rdm2file %s' % rdm2file.name)
        cmd.append('--nelec %s' % nelecas)
        cmd.append('--norb %s' % ncas)
        cmd.append('--wfnfile %s' % wfnfile)
        os.system(' '.join(cmd))
        with open(rdm1file.name, 'r') as f:
            dat = f.read()
            rdm1 = numpy.array(map(float, dat.split()))
            rdm1 = rdm1.reshape(ncas,ncas)
        with open(rdm2file.name, 'r') as f:
            dat = f.read()
            rdm2 = numpy.array(map(float, dat.split()))
            rdm2 = rdm2.reshape(ncas,ncas,ncas,ncas)
        return rdm1, rdm2

