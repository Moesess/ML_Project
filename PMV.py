import math

def calculate_pmv(ta, rh, tr=22, vel=0.1, met=1.2, clo=0.5, wme=0):
    """
    Oblicza wskaźnik PMV (Predicted Mean Vote) na podstawie podanych parametrów.

    :param ta: Temperatura powietrza w °C
    :param tr: Średnia temperatura promieniowania w °C
    :param rh: Wilgotność względna w %
    :param vel: Prędkość powietrza w m/s
    :param met: Metabolizm w metach (1 met = 58.2 W/m²)
    :param clo: Izolacja odzieży w clo (1 clo = 0.155 m²K/W)
    :param wme: Praca mechaniczna w metach (domyślnie 0)
    :return: Wartość PMV
    """

    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))
    icl = 0.155 * clo
    m = met * 58.2
    w = wme * 58.2
    mw = m - w
    if icl <= 0.078: fcl = 1 + 1.29 * icl
    else: fcl = 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(vel)
    taa = ta + 273
    tra = tr + 273
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * math.pow(tra/100, 4)
    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015
    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
        n += 1
        if n > 150:
            print("Nie udało się osiągnąć konwergencji w obliczeniach PMV")
            return None
    tcl = 100 * xn - 273
    hl1 = 3.05 * 0.001 * (5733 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15)
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - ta)
    hl5 = 3.96 * fcl * (math.pow(xf, 4) - math.pow(tra/100, 4))
    hl6 = fcl * hc * (tcl - ta)
    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    return pmv

def find_optimal_temperature(humidity, target_pmv=0, step=0.1, max_iterations=100):
    optimal = 20
    for _ in range(max_iterations):
        new_pmv = calculate_pmv(optimal, humidity)
        if abs(new_pmv - target_pmv) < 0.5:  # Tolerancja dla PMV
            return optimal

        if new_pmv > target_pmv:
            optimal -= step
        else:
            optimal += step

    return optimal
