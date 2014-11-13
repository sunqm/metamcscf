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
emc = mc.mc2step()[0] + mol.nuclear_repulsion()
print(ehf, emc, emc-ehf)

mc = mcscf.CASSCF(mol, mf, 4, 4)
emc = mc.mc1step()[0] + mol.nuclear_repulsion()
print(ehf, emc, emc-ehf)

