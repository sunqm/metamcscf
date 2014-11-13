import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
import cisolver

mol = gto.Mole()
mol.build(
    verbose = 4,
    output = None,
    atom = [['H', (0.,0.,i)] for i in range(8)],
    basis = {'H': 'sto-3g'},
)

mf = scf.RHF(mol)
ehf = mf.scf()
print(ehf)

mc = mcscf.CASSCF(mol, mf, 4, 4)
mc.fcisolver = cisolver.CISolver()
mc.max_orb_stepsize = .01 # max. orbital-rotation angle
mc.max_cycle_micro = 1    # small value for frequently call CI solver
mc.conv_threshold = 1e-4
def save_mo_coeff(mo_coeff, imacro, imicro):
    fname = 'mcscf-mo-%d-%d.npy' % (imacro+1, imicro+1)
    numpy.save(fname, mo_coeff)
mc.save_mo_coeff = save_mo_coeff
emc = mc.mc2step()[0] + mol.nuclear_repulsion()
print(ehf, emc, emc-ehf)

# restart from chkpoint of the 1st micro step of 2nd macro step
mc = mcscf.CASSCF(mol, mf, 4, 4)
mc.fcisolver = cisolver.CISolver()
mc.max_orb_stepsize = .05 # max. orbital-rotation angle
mc.max_cycle_micro = 3    # small value for frequently call CI solver
mc.conv_threshold = 1e-8
mo = numpy.load('mcscf-mo-3-1.npy')
emc = mc.mc2step(mo)[0] + mol.nuclear_repulsion()
print(ehf, emc, emc-ehf)

# reference results
mc = mcscf.CASSCF(mol, mf, 4, 4)
emc = mc.mc1step()[0] + mol.nuclear_repulsion()
print(ehf, emc, emc-ehf)

