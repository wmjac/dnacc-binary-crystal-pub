import argparse, gzip, pickle, math
import numpy as np

def build_binary_lattice(ax, ay, az, lx):
    a = np.array([ax, ay, az])
    atoms = np.zeros((2*lx**3, 3))
    atomtypes = np.zeros(2*lx**3)
    for x in range(lx):
        for y in range(lx):
            for z in range(lx):
                atoms[2*lx**2 * x + 2*lx * y + 2*z,:] = np.array([x, y, z]) * a
                atoms[2*lx**2 * x + 2*lx * y + 2*z + 1,:] = (np.array([x, y, z]) + 0.5) * a
                atomtypes[2*lx**2 * x + 2*lx * y + 2*z] = 1
                atomtypes[2*lx**2 * x + 2*lx * y + 2*z + 1] = 2
    box = lx * a
    return box, atoms, atomtypes

def create_nbr_list(atoms, box, rc):
    nbrs = {i : {} for i in range(atoms.shape[0])}
    for i in range(atoms.shape[0]):
        for j in range(i + 1, atoms.shape[0]):
            image = np.zeros(3)
            for d in range(3):
                if atoms[j,d] - atoms[i,d] < -box[d] / 2.:
                    image[d] = 1.
                elif atoms[j,d] - atoms[i,d] >= box[d] / 2.:
                    image[d] = -1.
            dr = atoms[j,:] - atoms[i,:] + image * box
            if np.linalg.norm(dr) <= rc:
                nbrs[i][j] = image
                nbrs[j][i] = -image
    return nbrs

def create_nbr_list_one(atoms, box, i, rc):
    nbrs = {0 : {}}
    for j in range(i + 1, atoms.shape[0]):
        image = np.zeros(3)
        for d in range(3):
            if atoms[j,d] - atoms[i,d] < -box[d] / 2.:
                image[d] = 1.
            elif atoms[j,d] - atoms[i,d] >= box[d] / 2.:
                image[d] = -1.
        dr = atoms[j,:] - atoms[i,:] + image * box
        if np.linalg.norm(dr) <= rc:
            nbrs[i][j] = image
    return nbrs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('Cspacing', type=float, help="spacing between C values")
    parser.add_argument('lx', type=int, help="lattice size")
    parser.add_argument('--cutoff', type=float, default=3., \
                        help="cutoff distance for neighbor calculation [3.]")
    clargs = parser.parse_args()

    print("# C rij_AA dS_harm num_nbrs")

    data = {}

    for C in np.linspace(0, 1, int(1. / clargs.Cspacing) + 1):

        req = 1.
        D = (2./math.sqrt(3.) * (1. - C) + math.sqrt(2.) * C) / (2./math.sqrt(3.) * (1. - C) + C)
        ax = ay = (2. / math.sqrt(2. + D**2)) * req
        az = D * ax
        box, atoms, atomtypes = build_binary_lattice(ax, ay, az, clargs.lx)
        rc = 1.16 * req

        nbrs = create_nbr_list(atoms, box, rc)
        bonds = {i : {j : nbrs[i][j] for j in nbrs[i] if atomtypes[i] != atomtypes[j]} for i in nbrs}

        i = 0
        for j in nbrs[i]:
            rij = atoms[j,:] - atoms[i,:] + nbrs[i][j] * box
            if j not in bonds[i]:
                rij_AA = np.linalg.norm(rij)
                break

        H = np.zeros((atoms.shape[0]*atoms.shape[1], atoms.shape[0]*atoms.shape[1]))
        nd = atoms.shape[1]
        for i in bonds:
            for j in bonds[i]:
                if i < j:
                    rij = atoms[j,:] - atoms[i,:] + nbrs[i][j] * box
                    rij /= np.linalg.norm(rij)
                    kij = 1.
                    for a in range(3):
                        for b in range(3):
                            H[i*nd + a, j*nd + b] -= kij * rij[a] * rij[b]
                            H[j*nd + b, j*nd + b] += kij * rij[b] * rij[b]
                            H[j*nd + b, i*nd + a] -= kij * rij[b] * rij[a]
                            H[i*nd + a, i*nd + a] += kij * rij[a] * rij[a]

        dS_harm = np.linalg.slogdet(H)[1] / (2. * atoms.shape[0])

        i = 0
        nbrs_i = create_nbr_list_one(atoms, box, i, clargs.cutoff)[i]
        nbr_dist = []
        nbr_like = []
        for j in nbrs_i:
            rij = atoms[j,:] - atoms[i,:] + nbrs_i[j] * box
            nbr_dist.append(np.linalg.norm(rij))
            nbr_like.append(atomtypes[i] == atomtypes[j])

        print("%g %g %g %d" % (C, rij_AA, dS_harm, len(nbr_dist)))

        data[C] = {'rij_AA' : rij_AA,
                   'dS_harm' : dS_harm,
                   'nbr_dist' : np.array(nbr_dist),
                   'nbr_like' : np.array(nbr_like)}

    with gzip.open('harmonic-BCT.p.gz', 'wb') as f:
        pickle.dump(data, f)
