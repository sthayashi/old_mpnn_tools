import numpy as np

import h5py

f_v1 = h5py.File("v1.h5", 'r')
f_v2 = h5py.File("v2.h5", 'r')
f_v3 = h5py.File("v2.h5", 'r')
f_serial = h5py.File("serial.h5", 'r')

mol_v1 = f_v1.get(list(f_v1.keys())[0])
mol_v2 = f_v2.get(list(f_v2.keys())[0])
mol_v3 = f_v3.get(list(f_v3.keys())[0])
mol_serial = f_serial.get(list(f_serial.keys())[0])

ti_v1 = mol_v1.get("transfer_integrals")[:]
ti_v2 = mol_v2.get("transfer_integrals")[:]
ti_v3 = mol_v3.get("transfer_integrals")[:]
ti_serial = mol_serial.get("transfer_integrals")[:]

inds_v1 = mol_v1.get("o2_good_inds")[:]
inds_v2 = mol_v2.get("o2_good_inds")[:]
inds_v3 = mol_v3.get("o2_good_inds")[:]
inds_serial = mol_serial.get("o2_good_inds")[:]


print("v1 and v2")
print(np.alltrue(ti_v1[inds_v1] == ti_v2[inds_v2]))

print("v1 and v3")
print(np.alltrue(ti_v1[inds_v1] == ti_v3[inds_v3]))

print("v1 and serial")
print(np.alltrue(ti_v1[inds_v1] == ti_serial[inds_serial]))
