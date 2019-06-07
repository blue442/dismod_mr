import numpy as np
from scipy import integrate
import pdb


def runge_kutta_4(f, ti, yi, dt):
    pdb.set_trace()
    k1 = dt * f(ti, yi)
    k2 = dt * f(ti + .5*dt, yi + .5*k1)
    k3 = dt * f(ti + .5*dt, yi + .5*k2)
    k4 = dt * f(ti + dt, yi + k3)
    yf = yi + (1./6.) * (k1 + 2.*k2 + 2.*k3 + k4)
    return yf


m_all = np.array([0.04450774, 0.03636661, 0.02008435, 0.0038021, 0.00313581,
                  0.00246952, 0.00180323, 0.00113694, 0.00078469, 0.00074649,
                  0.0007083, 0.0006701, 0.0006319, 0.00067136, 0.00078849,
                  0.00090562, 0.00102274, 0.00113987, 0.00127202, 0.0014192,
                  0.00156638, 0.00171356, 0.00186074, 0.00198843, 0.00209663,
                  0.00220483, 0.00231304, 0.00242124, 0.00252962, 0.00263818,
                  0.00274673, 0.00285529, 0.00296385, 0.00310445, 0.00327712,
                  0.00344978, 0.00362244, 0.0037951, 0.00397422, 0.00415979,
                  0.00434537, 0.00453095, 0.00471652, 0.00497815, 0.00531584,
                  0.00565353, 0.00599122, 0.00632891, 0.00677505, 0.00732963,
                  0.00788421, 0.00843879, 0.00899337, 0.00965486, 0.01042326,
                  0.01119165, 0.01196005, 0.01272844, 0.01378992, 0.01514449,
                  0.01649906, 0.01785362, 0.01920819, 0.02081546, 0.02267544,
                  0.02453542, 0.0263954, 0.02825538, 0.03074407, 0.03386148,
                  0.0369789, 0.04009631, 0.04321372, 0.04706369, 0.05164623,
                  0.05622877, 0.0608113, 0.06539384, 0.07114923, 0.07807748,
                  0.08500573, 0.09193398, 0.09886223, 0.10849057, 0.12081899,
                  0.13314741, 0.14547584, 0.15780426, 0.17297873, 0.19099924,
                  0.20901975, 0.22704025, 0.24506076, 0.26572334, 0.28902798,
                  0.31233262, 0.33563725, 0.35894189, 0.39975041, 0.45806281,
                  0.51637521])
N = len(m_all)
num_step = 10  # double until it works

ages = np.array([0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
                 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
                 22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,  32.,
                 33.,  34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42.,  43.,
                 44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.,  52.,  53.,  54.,
                 55.,  56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.,  64.,  65.,
                 66.,  67.,  68.,  69.,  70.,  71.,  72.,  73.,  74.,  75.,  76.,
                 77.,  78.,  79.,  80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,
                 88.,  89.,  90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,
                 99., 100.])

num_step = num_step
age_local = ages
all_local = m_all

N = len(age_local)
age = age_local
all_cause = all_local
susceptible = np.zeros(N)
condition = np.zeros(N)
incidence = np.zeros(N)
remission = np.zeros(N)
excess = np.zeros(N)
s0 = 0.
c0 = 0.


def ode_fun(w, a, p):
    y1, y2 = w
    i, m, e, r = p

    dyda = [-(i + (m - e*y1) / (y1+y2)) * y1 + r*y2,
            i * y1 - (r + (m - e*y1) / (y1+y2) + e) * y2]

    return dyda

    global age, incidence, remission, excess, all_cause

    i = incidence[idx]
    r = remission[idx]
    e = excess[idx]
    m = all_cause[idx]


def ode_fun(susceptible_condition, a):
    global age, incidence, remission, excess, all_cause
    s = susceptible_condition[0]
    c = susceptible_condition[1]

    # APC: added this in order to find the index (which needed to be an int instead of a float)
    idx = int(np.argwhere(age == a))

    i = incidence[idx]
    r = remission[idx]
    e = excess[idx]
    m = all_cause[idx]

    other = m - e * s / (s + c)
    ds_da = - (i + other) * s + r * c
    dc_da = +           i * s - (r + other + e) * c
    return np.array([ds_da, dc_da])


def ode_integrate(N, num_step, s0, c0):
    global age, incidence, remission, excess, all_cause
    global susceptible, condition
    susceptible[0] = s0
    condition[0] = c0
    sc = np.array([s0, c0])
    N = len(all_cause)

    for j in range(N-1):
        # a_step = (age[j+1] - age[j]) / num_step
        # a_tmp = age[j]

        sc = integrate.odeint(func=ode_fun,
                              t=np.linspace(start=age[j], stop=age[j+1], num=num_step),
                              y0=sc, tfirst=True)

        # for step in range(num_step):
        #     pdb.set_trace()
        #     sc = runge_kutta_4(ode_fun, a_tmp, sc, a_step)
        #     a_tmp = a_tmp + a_step
        susceptible[j+1] = sc[0]
        condition[j+1] = sc[1]


# x = np.hstack((incidence, remission, excess, s0, c0))
# # x = pycppad.independent(x)
# incidence = x[(0*N):(1*N)]
# remission = x[(1*N):(2*N)]
# excess = x[(2*N):(3*N)]
# s0 = x[3*N]
# c0 = x[3*N+1]


aaa = ode_integrate(N, num_step, s0, c0)
# y = np.hstack((susceptible, condition))
# fun = pycppad.adfun(x, y)
# fun = dismod_mr.model.ode.ode_function(num_step, ages, m_all)
