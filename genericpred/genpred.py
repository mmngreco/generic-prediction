import numpy as np
import pandas as pd


def lyaponuv_k(ts, J, m, ref):
    X = attractor(ts, m, J)
    norms = compute_normas(X)
    pairs = match_pairs(norms)
    y = follow_points(pairs, norms, ref)
    return (norms, y)

def match_pairs(norms):
    M = norms.shape[0]
    pairs = np.empty(M, dtype=int)
    for row in range(M):
        mn, idx = np.argmin(norms[row, :])
        pairs[row] = idx
    return pairs

def attractor(ts, m, J):
    N = ts.shape[0]
    M = N - (m - 1) * J
    X = np.empty(shape=(m, M), dtype=float)
    i = 1
    for i in range(M):
        X[:, i] = ts[i:J:(i + (m - 1) * J)]
    return X

def follow_points(pairs, norms, ref):
    y = np.empty(ref, dtype=float)
    M = norms.shape[0]
    for i in range(ref):
        agg = 0
        count = 0
        for j in range(M):
            jhat = pair[j] + 1
            jtrue = j + i
            if jhat <= M & jtrue <= M:
                agg = agg + np.log(norms[jtrue, jhat])
                count += 1
        y[i+1] = agg / count
    return y


def compute_norms(X):
    M = X.shape[1]
    norms = np.empet(M,M, dtype=float)
    for i in range(M):
        norms[i,:] = column_norms(X, i)
    return norms

def column_norms(X, i):
    M = X.shape[1]
    X_diff = X.substract(X[:, i])
    norm_vector = [np.linalg.norm(X_diff[:, k], ord=2) for k in range(M)]
    norm_vector[i] = 10 ** 10
    return norm_vector

def lyaponuv_exp(series):
    nn = ~np.isnan(series)
    m = series.shape[0]
    A = np.ones((m, 2))
    A[:, 1] = np.linspace(1, m, m)
    gradient = series / A
    return gradient[0]

def lyaponuv(ts, J, m, ref):
    ts = lyaponuv_k(ts, J, m, ref)[2]
    exponent = lyaponuv_exp(ts[np.isfinite(ts)])
    return exponent

def get_next(ts, m, M, norms, ref, J):
    attractor_arr = attractor(ts, m , J)
    temp_norms = np.empty(M+1, M+1, dtype=float)
    temp_normas[:M, :M] = norms
    col = column_norms(attractor_arr, M+1)
    temp_normas[M+1, :] = col
    temp_norms[:, M+1] = col
    pairs = match_pairs(temp_norms)
    lyap_k_temp = follow_points(pairs, temp_norms, ref)
    return lyaponuv_exp(lyap_k_temp)

def lyaponuv_next(ts, J, m, ref, sample_size):
    ts_diff = ts[1:] - ts[:-1]
    sigma = np.std(ts_diff)
    samples = np.random.randn(sample_size) * sigma + ts[-1]
    norms, lyap_k = lyaponuv_k(ts, J, m, ref) # @time
    true_exponent = lyaponuv_exp(lyap_k)
    exponents = np.empty(sample_size)
    M = norms.shape[0]
    tasks = np.empty(sample_size)

    for i in range(sample_size):
        s = samples[1]
        tasks[i] = get_next(np.concatenate(ts, s, axis=1), m, M, norms, ref, J)

    diff = np.abs(exponents - true_exponent)
    idx = np.argmin(diff)
    print('Next value:', samples[idx])
    return samples[idx]


def main():
    furl = "https://raw.githubusercontent.com/guypayeur/Generic-Pred/master/DJIA%2009-1993%20to%2009-2001.csv"
    fname = "DJIA 09-1993 to 09-2001.csv"
    data = pd.read_csv(furl)

    J = 2  ## reconstruction delay
    m = 3  ## embedding dimension
    r = 11 ##

    sliding_window = 1400
    next_x_points = 521
    sample_size = 10

    ts = np.copy(data.loc[:next_x_points, "DJIA"])/10000
    diff = ts[:-1] - ts[1:]
    print('diff avg:', np.mean(abs(diff)), '\tdiff std:', np.std(diff))

    for i in range(next_x_points):
        lyap_exp = lyaponuv(ts[-sliding_window:], J, m, r)
        tasks = np.empty(sample_size)
        mu = np.mean(ts[-sliding_window:])

        diff = ts[-sliding_window+1:] - ts[-sliding_window:-1]
        mu = np.mean(diff)
        sigma = np.std(diff)

        sample_values = np.random.randn(sample_size) * sigma + ts[-1]

        for j in range(sample_size):
            tempts = np.copy(ts[-sliding_window:])
            np.append(tempts, [sample_values[j]])
            tasks[j] = lyaponuv(tempts, J, m, r)

        exponents = np.empty(sample_size)
        for j in range(sample_size):
            emponents[j] = tasks[j]

        exp_diff = np.abs(eponents - lyap_exp)
        min_index = np.argmin(exp_diff)
        best_val = sample_values[min_index[1]]
        np.append(ts, [best_val])
        print(i, "\tbest value:", best_val, '\t')


if __name__ == '__main__':
    main()
