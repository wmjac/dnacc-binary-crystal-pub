import math, argparse, gzip, pickle
import numpy as np
import scipy.interpolate, scipy.optimize, scipy.integrate

def define_fluid(epsilon, delta_rel, dmu_fluid):
    def mu_fluid(log_eta):
        eta = math.exp(log_eta)
        return (log_eta \
                + (8.*eta - 9.*eta**2 + 3.*eta**3) / (1. - eta)**3 \
                + 8. * epsilon * ((1. + 0.5 * delta_rel)**3 - (1. - 0.5 * delta_rel)**3) * eta \
                + dmu_fluid * eta)
    def P_fluid(log_eta):
        eta = math.exp(log_eta)
        return ((eta + eta**2 + eta**3 - eta**4) / (1. - eta)**3 \
                + 4. * epsilon * ((1. + 0.5 * delta_rel)**3 - (1. - 0.5 * delta_rel)**3) * eta**2 \
                + 0.5 * dmu_fluid * eta**2)
    return mu_fluid, P_fluid

def define_crystal(f, eta, dmu_crystal):
    def mu_crystal(Pv):
        return (f + Pv / eta + dmu_crystal)
    return mu_crystal

def calc_coex(epsilon, delta_rel, f_crystal, eta_crystal, dmu_fluid, dmu_crystal):
    mu_fluid, Pv_fluid = define_fluid(epsilon, delta_rel, dmu_fluid)
    mu_crystal = define_crystal(f_crystal, eta_crystal, dmu_crystal)
    def delta_mu(log_eta_fluid):
        return mu_fluid(log_eta_fluid) - mu_crystal(Pv_fluid(log_eta_fluid))
    log_eta_fluid = scipy.optimize.root_scalar(delta_mu, bracket=(-6000., math.log(0.73))).root
    return log_eta_fluid, mu_fluid(log_eta_fluid), Pv_fluid(log_eta_fluid)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('potential', type=str, help="path to potential file")
    parser.add_argument('sigma_colloid', type=float, help="colloid diameter in nm")
    parser.add_argument('--include-like-attraction', action='store_true', \
                        help="include attraction between like particles in calculations [True]")
    parser.add_argument('--exclude-like-repulsion', action='store_true', \
                        help="exclude repulsion between like particles in calculations [False]")
    parser.add_argument('--exclude-vdW', action='store_true', \
                        help="exclude vdW contribution in calculations [False]")
    parser.add_argument('--print-pair-potential', action='store_true', \
                        help="only print pair potential [False]")
    clargs = parser.parse_args()

    # Read the pair potential:
    with open(clargs.potential, 'r') as f:
        # h (in meters), phi_DNA_binding, phi_Steric, phi_Vdw, [phi_DNA_binding_like (if included)]
        data = np.array([[float(y) for y in line.split()] \
                         for line in f if len(line) > 0 and line[0] != '#'])
    # # NOTE: Excluding electrostatic contribution (which is negligible) in the following potentials.
    r_discrete = np.array([data[i,0]*1.e9 + clargs.sigma_colloid for i in range(data.shape[0])])
    u_AB_discrete = np.array([(data[i,1] + data[i,2]) for i in range(data.shape[0])])
    u_AB = scipy.interpolate.CubicSpline(r_discrete, u_AB_discrete)
    if clargs.include_like_attraction:
        u_AA_discrete = np.array([(data[i,2] + data[i,4]) for i in range(data.shape[0])])
        u_AA = scipy.interpolate.CubicSpline(r_discrete, u_AA_discrete)
    else:
        u_AA_discrete = np.array([(data[i,2]) for i in range(data.shape[0])])
        u_AA = scipy.interpolate.CubicSpline(r_discrete, u_AA_discrete)
    u_vdW_discrete = np.array([(data[i,3]) for i in range(data.shape[0])])
    u_vdW = scipy.interpolate.CubicSpline(r_discrete, u_vdW_discrete)

    sigma = scipy.optimize.minimize_scalar(u_AB, (r_discrete[np.argmin(u_AB_discrete)] - 1., \
                                                  r_discrete[np.argmin(u_AB_discrete)] + 1.)).x
    epsilon = u_AB(sigma)
    kAB = u_AB(sigma, 2) * sigma**2
    delta_rel = math.sqrt(8. / kAB)
    zAB, zAA = 8., 4.
    print("# sigma =", sigma)
    print("# epsilon =", epsilon)
    print("# k_AB=", kAB)
    print("# delta/sigma =", delta_rel)

    if clargs.include_like_attraction:
        sigma_AA = scipy.optimize.minimize_scalar(u_AA, (r_discrete[np.argmin(u_AA_discrete)] - 1., \
                                                         r_discrete[np.argmin(u_AA_discrete)] + 1.)).x
        epsilon_AA = u_AA(sigma_AA)
        u_rep = lambda r: u_AA(r) - epsilon_AA if r <= sigma_AA else 0.
        u_att = lambda r: epsilon_AA if r <= sigma_AA else u_AA(r)
    else:
        u_rep = u_AA
        u_att = lambda r: 0.

    if clargs.print_pair_potential:
        with open('pair-potential.dat', 'w') as f:
            f.write("# r r/sigma u_AB u_AA u_rep u_like_att\n")
            for r in np.linspace(clargs.sigma_colloid * 0.8, clargs.sigma_colloid * 1.6, 2000):
                f.write("%g %g %g %g %g %g\n" % (r, r / sigma, u_AB(r), u_AA(r), u_rep(r), u_att(r)))
        raise SystemExit

    # Load the BCT data:
    with gzip.open('harmonic-BCT.p.gz', 'rb') as f:
        bctdata = pickle.load(f)

    # Compute coexistence for each C value:
    mu_coex, eta_fluid_coex = {}, {}
    print("\n# %4s %10s %10s %10s %10s %10s %10s %10s %10s %10s" % \
          ('C', 'mu', 'mu-mu(0)', 'log(Pv)', 'log(eta_fl)', 'eta_c', 'f_c', \
           'dmu_rep_c', 'dmu_vdW_c', 'dmu_like_c'))
    for C in sorted(bctdata):

        # Define the crystal parameters:
        D = (2./math.sqrt(3.) * (1. - C) + math.sqrt(2.) * C) / (2./math.sqrt(3.) * (1. - C) + C)
        ax = (2. / math.sqrt(2. + D**2)) * sigma # = ay
        az = D * ax
        eta_crystal = (math.pi / 6. * sigma**3) / (0.5 * ax**2 * az)

        # Define the reference crystal free energy:
        dS_harm = -1.5 * math.log(2. * math.pi / kAB) + bctdata[C]['dS_harm']
        dS_symm = -math.log(2.) # Assume ordered binary crystal.
        f_crystal = 0.5 * zAB * epsilon + dS_harm + dS_symm

        # Compute corrections to the reference model:
        rij_AA = bctdata[C]['rij_AA']
        kAA = kAB * 1.e-4
        if not clargs.exclude_like_repulsion:
            # ra, rb = 0., rij_AA
            ra, rb = 0., rij_AA
            g_AA = lambda r: math.exp(-0.5 * kAA * (r - rij_AA)**2)
            Vex = scipy.integrate.quad(lambda r: g_AA(r) * math.exp(-u_rep(r * sigma)), ra, rb)[0]
            Vtot = scipy.integrate.quad(lambda r: g_AA(r), ra, rb)[0]
            try:
                dmu_rep_crystal = -math.log(Vex / Vtot) # = -dS_anharmonic
            except (ValueError, ZeroDivisionError):
                continue
        else:
            dmu_rep_crystal = 0.
        if not clargs.exclude_vdW:
            dmu_vdW_fluid = \
                24. * scipy.integrate.quad(lambda r: r**2 * u_vdW(r), sigma, np.inf)[0] / sigma**3
            dmu_vdW_crystal = sum(u_vdW(sigma * rij) for rij in bctdata[C]['nbr_dist'])
        else:
            dmu_vdW_fluid = dmu_vdW_crystal = 0.
        if clargs.include_like_attraction:
            if clargs.exclude_like_repulsion: raise Exception
            dmu_att_fluid = \
                12. * scipy.integrate.quad(lambda r: r**2 * u_att(r), sigma, np.inf)[0] / sigma**3
            ra, rb = 0., np.inf # 2. * rij_AA
            u_att_avg = scipy.integrate.quad(lambda r: g_AA(r) * u_att(sigma * r), ra, rb)[0]
            Vtot = scipy.integrate.quad(lambda r: g_AA(r), ra, rb)[0]
            dmu_att_crystal = zAA * u_att_avg / Vtot
        else:
            dmu_att_fluid = dmu_att_crystal = 0.

        # Compute coexistence:
        try:
            log_eta_fluid, mu, Pv = calc_coex(epsilon, delta_rel, f_crystal, eta_crystal,
                                              dmu_vdW_fluid + dmu_att_fluid, \
                                              dmu_rep_crystal + dmu_vdW_crystal + dmu_att_crystal)
        except ValueError as e:
            print(e)
            continue
        try:
            log_Pv = math.log(Pv)
        except ValueError:
            log_Pv = -np.inf
        mu_coex[C] = mu
        eta_fluid_coex[C] = math.exp(log_eta_fluid)

        print("%6.4f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f" % \
              (C, mu, mu - mu_coex[0], log_Pv, log_eta_fluid, eta_crystal, f_crystal, \
               eta_crystal * dmu_rep_crystal, eta_crystal * dmu_vdW_crystal, \
               eta_crystal * dmu_att_crystal))

    C_eq = min(mu_coex, key=lambda C: mu_coex[C])
    mu_coex_eq = mu_coex[C_eq]
    eta_fluid_coex_eq = eta_fluid_coex[C_eq]

    print("\n# C_eq = %g ; eta_fluid = %g" % (C_eq, eta_fluid_coex_eq))
