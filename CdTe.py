from analyse import *
import matplotlib.pyplot as mpl


def main ():
    data, popts, pcovs, Lambda, Resolution = analyse (1620, 1600, "am_241 CdTe.mca")

    print ("Comptes totaux:")
    [print (f"\tPic {i+1}: {lambda_i} ± {np.sqrt (lambda_i)}") for lambda_i, i in zip (Lambda, range (3))]
    print ("Résolutions relatives:")
    [print (f"\tPic {i+1}: {res[0]} ± {res[1]}") for res, i in zip (Resolution, range (3))]

    cannaux = np.linspace (1, 8192, 8192, dtype=int)
    mpl.figure (layout="constrained")
    mpl.plot (cannaux, data, ".", label="Compte")
    linestyle = ["-", "--", "-."]
    for i in range (3):
        popt = popts[i]
        pstd = np.sqrt (np.diag (pcovs[i]))

        A = popt[0]
        sigmaA = pstd[0]
        mu = popt[1]
        sigmaMu = pstd[1]
        sigma = popt[2]
        sigmaSigma = pstd[2]

        gauss = partial (gaussienne, A=A, mu=mu, sigma=sigma)
        # Intervalle de confiance à 95%
        gaussMax = partial (gaussienne, A=A+3*sigmaA, mu=mu, sigma=sigma+3*sigmaSigma)
        gaussMin = partial (gaussienne, A=A-3*sigmaA, mu=mu, sigma=sigma-3*sigmaSigma)

        x = np.linspace (mu-4*sigma, mu+4*sigma, 100)
        p = mpl.plot (x, gauss (x), linestyle[i], label=fr"Pic {i+1}, $A = {A:.3}\pm{sigmaA:.1}$, $\mu = {mu:.5}\pm{sigmaMu:.1}$, $\sigma = {sigma:.3}\pm{sigmaSigma:.1}$")
        mpl.fill_between (x, gaussMin (x), gaussMax (x), alpha=0.3, color=p[0].get_c ())

    mpl.xlabel ("Cannal")
    mpl.ylabel ("Compte")
    mpl.title ("Régressions de compte en fonction du cannal pour les photopics\nde 13.95, 17.74 et 59.54 keV pour le 241Am avec un détecteur CdTe.")
    mpl.minorticks_on ()
    mpl.tick_params (which="both", direction="in")
    mpl.legend ()
    mpl.show ()


if __name__ == "__main__":
    main ()
