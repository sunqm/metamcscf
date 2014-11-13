#!/bin/bash

while test -n "$1"; do
  case "$1" in
    --fcidump)
      fcidump=$2
      shift 2
      ;;  
    --wfnfile)
      wfnfile=$2
      shift 2
      ;;  
    --rdm1file)
      rdm1file=$2
      shift 2
      ;;  
    --rdm2file)
      rdm2file=$2
      shift 2
      ;;  
    --norb)
      norb=$2
      shift 2
      ;;  
    --nelec)
      nelec=$2
      shift 2
      ;;  
    *)
      break
      ;;  
  esac
done 

if [[ -n $fcidump ]]; then
# compute wfn
  python << EOF
import numpy
import pyscf.lib
import pyscf.ao2mo
import pyscf.fci
import os
with open('$fcidump', 'r') as f:
    dat = f.readline()
    for k in dat.split(','):
        if 'NORB' in k:
            norb = int(k.split('=')[1])
        elif 'NELEC' in k:
            nelec = int(k.split('=')[1])
    while 'END' not in dat:
        dat = f.readline()
    npair = norb*(norb+1)/2
    h1e = numpy.zeros(norb*(norb+1)/2)
    eri = numpy.zeros(npair*(npair+1)/2)
    dat = f.readline()
    while dat:
        c = dat.split()
        i,j,k,l = [int(x)-1 for x in c[1:]]
        if k != -1:
            if i >= j:
                ij = i*(i+1)/2+j
            else:
                ij = j*(j+1)/2+i
            if k >= l:
                kl = k*(k+1)/2+l
            else:
                kl = l*(l+1)/2+k
            if ij >= kl:
                ijkl = ij*(ij+1)/2 + kl
            else:
                ijkl = kl*(kl+1)/2 + ij
            eri[ijkl] = float(c[0])
        elif i != -1:
            if i >= j:
                ij = i*(i+1)/2+j
            else:
                ij = j*(j+1)/2+i
            h1e[ij] = float(c[0])
        else:
            nuc = float(c[0])
        dat = f.readline()
h1e = pyscf.lib.unpack_tril(h1e)
eri = pyscf.ao2mo.restore(1, eri, norb)
e, ci = pyscf.fci.direct_spin0.kernel(h1e, eri, norb, nelec)

with open('$wfnfile', 'w') as f:
    for x in ci.ravel():
        f.write('%20.15g\n' % x)
print('%20.15f'%e)
EOF

else

# compute density matrix
  python << EOF
import numpy
import pyscf.ao2mo
import pyscf.fci
norb = $norb
nelec = $nelec
with open('$wfnfile', 'r') as f:
    dat = map(float, f.read().split())
    wfn = numpy.array(dat)
rdm1, rdm2 = pyscf.fci.direct_spin0.make_rdm12(wfn, norb, nelec)

with open('$rdm1file', 'w') as f:
    for x in rdm1.ravel():
        f.write('%20.15g\n' % x)
with open('$rdm2file', 'w') as f:
    for x in rdm2.ravel():
        f.write('%20.15g\n' % x)
EOF

fi

