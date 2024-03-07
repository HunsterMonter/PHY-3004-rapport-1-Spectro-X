from functools import partial
from scipy.optimize import curve_fit
import matplotlib.pyplot as mpl
import numpy as np
import numpy.ma as ma


gaussienne = lambda x, A, mu, sigma: A*np.exp (-((x-mu)/sigma)**2/2)


def analyse (cutoff: int, nouveauGauche: int, fichier: str) -> tuple[np.ndarray[int], list[list[float]], list[list[list[float]]], list[float], list[list[float]]]:
    """
    Fonction qui permet d'analyser les trois photopics à 13.95, 17.74 et 59.54 keV du 241Am

    Params:
        cutoff: Cutoff pour le fit de la gaussienne qui permet d'éliminer le photopic
            parasite tout juste à gauche du deuxième photopic d'intérêt
        nouveauGauche: Déplace la borne gauche de la région d'intérêt du deuxième
            photopic à cette valeur après avoir retiré le photopic parasite
        fichier: Nom du fichier à analyser

    Returns:
        data: Array numpy contenant le compte en fonction du canal, ne contient
            que des données non nulles où sont les photopics
        popts: Liste des paramètres optimaux pour les gaussiennes des photopics
        pcovs: Liste des matrices de covariances pour les paramètres optimaux
        Lambda: Liste du compte total pour chaque photopic
        Resolution: Liste de la résolution relative et de sont incertitude
            pour chaque photopic
    """
    # Comptes des canaux
    data: np.ndarray[int] = np.loadtxt (fichier, skiprows=16, max_rows=8192, dtype=int)
    # Régions d'intérêt
    roi: np.ndarray[np.ndarray[int]] = np.loadtxt (fichier, skiprows=12, max_rows=3, dtype=int)

    # On masque toutes les données, elles seront démasquées lors de l'analyse
    data = ma.array (data, mask=True)

    # Arrays qui sont retournés contenant les informations pour chaque photopic analysé
    popts: list[list[float]] = []
    pcovs: list[list[list[float]]] = []
    Lambda: list[float] = []
    Resolution: list[list[float]] = []

    # On analyse les trois pics
    for region, i in zip (roi, range (3)):
        # On définit la gauche et la droite du photopic et on le démasque
        gauche: float = region[0]
        droite: float = region[1]
        data.mask[gauche:droite+1] = False

        # Pour le deuxième pic, on doit retirer le photopic parasitif adjacent
        if i == 1:
            # On calcule le centre, la largeur et l'amplitude estimée de la région d'intérêt
            centre: float = (cutoff+gauche)/2
            largeur: float = cutoff-gauche
            amplitude: float = data[int (centre)]

            # Région sur laquelle fit la gaussienne à retirer et fit
            x: np.ndarray[int] = np.linspace (gauche, cutoff, largeur+1, dtype=int)
            rmpopt, rmpcov = curve_fit (gaussienne, x, data[gauche:cutoff+1], p0=(amplitude, centre, largeur/5))

            # On définit la gaussienne à retirer et on retire sa valeur de tous les canaux
            deGauss = partial (gaussienne, A=rmpopt[0], mu=rmpopt[1], sigma=rmpopt[2])
            x: np.ndarray[int] = np.linspace (gauche, droite, droite-gauche+1, dtype=int)
            data[gauche:droite+1] -= np.rint (deGauss (x)).astype (int)
            # Si une valeur est négative, on la met à 0
            data[gauche:droite+1] = np.where (data[gauche:droite+1] < 0, 0, data[gauche:droite+1])

            # On masque la portion utilisée pour retirer la gaussienne qui n'est pas inclue dans le pic
            data.mask[gauche:nouveauGauche] = True
            # La borne gauche est déplacée
            gauche = nouveauGauche

        # On calcule le centre et la largeur de la région d'intérêt
        # Si i == 1, le calcul est pour la région rapetissée
        centre: float = (droite+gauche)/2
        largeur: float = droite-gauche

        # On retire les points où le compte est nul, car curve_fit throw sinon
        intervalle: np.ndarray[int] = data[gauche:droite+1]
        x: np.ndarray[int] = np.nonzero (intervalle)[0] + gauche
        intervalle = intervalle [intervalle != 0]
        
        # Bornes pour l'optimisation des paramètres
        bounds: tuple[list[float]] = ([0, gauche, 1], [2*intervalle.max (), droite, largeur])
        # Fit du photopic, l'écart type sur le compte est obtenu en tenant compte que la variance
        # de la distribution de poisson est lambda, qui est estimé par le compte du canal
        popt, pcov = curve_fit (gaussienne, x, intervalle, p0=(intervalle.max (), centre, largeur/5), sigma=np.sqrt(intervalle))
        popts.append (popt)
        pcovs.append (pcov)

        # On somme tous les comptes de l'intervalle pour obtenir le compte total
        Lambda.append (intervalle.sum ())

        # Calcul de la résolution absolue et de son incertitude
        mu: float = popt[1]
        sigma2mu: float = pcov[1][1]
        sigma: float = popt[2]
        sigma2sigma: float = pcov[2][2]
        covSigmaMu: float = pcov[1][2]

        Res: float = 2.35*sigma/mu
        sigmaRes: float = Res * np.sqrt (sigma2sigma/sigma**2 + sigma2mu/mu**2 - 2*covSigmaMu/(sigma*mu))
        Resolution.append ((Res, sigmaRes))

    # Met à zéro les valeurs qui sont encore masquées
    data = data.filled (0)

    return data, popts, pcovs, Lambda, Resolution


def main ():
    data, poptsSiPIN, pcovsSiPIN, LambdaSiPIN, ResolutionSiPIN = analyse (1830, 1820, "am_241 Si-PIN.mca")
    data, poptsCdTe, pcovsCdTe, LambdaCdTe, ResolutionCdTe = analyse (1620, 1600, "am_241 CdTe.mca")

    # Hauteurs relatives entre le pic 1 et le pic 3 pour les deux détecteurs
    A1SiPIN = poptsSiPIN[0][0]
    sigma2A1SiPIN = pcovsSiPIN[0][0][0]
    A3SiPIN = poptsSiPIN[2][0]
    sigma2A3SiPIN = pcovsSiPIN[2][0][0]

    alphaSiPIN = A1SiPIN/A3SiPIN
    sigmaAlphaSiPIN = alphaSiPIN * np.sqrt (sigma2A1SiPIN/A1SiPIN**2 + sigma2A3SiPIN/A3SiPIN**2)

    A1CdTe = poptsCdTe[0][0]
    sigma2A1CdTe = pcovsCdTe[0][0][0]
    A3CdTe = poptsCdTe[2][0]
    sigma2A3CdTe = pcovsCdTe[2][0][0]

    alphaCdTe = A1CdTe/A3CdTe
    sigmaAlphaCdTe = alphaCdTe * np.sqrt (sigma2A1CdTe/A1CdTe**2 + sigma2A3CdTe/A3CdTe**2)

    print ("Comparaison de alpha:")
    print (f"\tSi-PIN: {alphaSiPIN:.5f} ± {3*sigmaAlphaSiPIN:.5f}")
    print (f"\tCdTe:   {alphaCdTe:.5f} ± {3*sigmaAlphaCdTe:.5f}")

    # On divise le compte de chaque pic par la probabilité d'émission du
    # pic pour normaliser les résultats, de plus, on multiplie le compte
    # du détecteur CdTe par 1/(1-0.0108) pour compenser le temps mort de 1.08%
    raies: list[float] = [13.95, 17.74, 59.54]
    coeffs: list[float] = [0.1160, 0.1183, 0.359]
    sigmacoeffs: list[float] = [0.0012, 0.0012, 0.004]

    effSiPIN = [lambda_i/c for lambda_i, c in zip (LambdaSiPIN, coeffs)]
    sigmaeffSiPIN = [lambda_i/c * np.sqrt (1/lambda_i + sigmac**2/c**2) for lambda_i, c, sigmac in zip (LambdaSiPIN, coeffs, sigmacoeffs)]

    effCdTe = [2/3 * lambda_i/(c*(1-0.0108)) for lambda_i, c in zip (LambdaCdTe, coeffs)]
    sigmaeffCdTe = [2/3 * lambda_i/(c*(1-0.0108)) * np.sqrt (1/lambda_i + sigmac**2/c**2) for lambda_i, c, sigmac in zip (LambdaCdTe, coeffs, sigmacoeffs)]

    mpl.figure (layout="constrained", figsize=(4, 3))
    mpl.errorbar (raies, [x/max (effCdTe) for x in effSiPIN], yerr=[3*x/max (effCdTe) for x in sigmaeffSiPIN], fmt=".", capsize=4, label="Si-PIN")
    mpl.errorbar (raies, [x/max (effCdTe) for x in effCdTe], yerr=[3*x/max (effCdTe) for x in sigmaeffCdTe], fmt="^", markersize=4, capsize=4, label="CdTe")
    mpl.xlabel ("Énergie [keV]")
    mpl.ylabel ("Efficacité relative normalisée")
    mpl.xscale ("log")
    mpl.yscale ("log")
    mpl.minorticks_on ()
    mpl.tick_params (which="both", direction="in")
    mpl.legend ()
    mpl.show ()

    mpl.figure (layout="constrained", figsize=(4, 3))
    mpl.errorbar (raies, [res[0] for res in ResolutionSiPIN], yerr=[3*res[1] for res in ResolutionSiPIN], capsize=4, fmt=".", label="Si-PIN")
    mpl.errorbar (raies, [res[0] for res in ResolutionCdTe], yerr=[3*res[1] for res in ResolutionCdTe], capsize=4, fmt="^", markersize=4, label="CdTe")
    mpl.xlabel ("Énergie [keV]")
    mpl.ylabel ("Résolution relative")
    mpl.minorticks_on ()
    mpl.tick_params (which="both", direction="in")
    mpl.legend ()
    mpl.show ()


if __name__ == "__main__":
    main ()
