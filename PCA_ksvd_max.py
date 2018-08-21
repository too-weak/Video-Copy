def PCA_ksvd_max(inpath, outpath, fps, meanX, coef, whiten, U, G):
    import readData
    import numpy as np
    from sklearn.linear_model import OrthogonalMatchingPursuit
    secPerBlock = 1
    overlap = 0
    block = fps * secPerBlock / 10
    step = block * (1-overlap)
    nonzero = 32

    A = readData.dlmread(inpath)

    # l2 norm
    for i in range(A.size[0]):
        A[i] /= np.linalg.norm(A[i])

    # pca
    A = A - meanX
    SCORE = np.matmul(A, coef)
    SCORE /= whiten

    # l2 norm
    for i in range(SCORE.size[0]):
        SCORE[i] /= np.linalg.norm(SCORE[i])

    # sparse
    A = np.transpose(SCORE)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=nonzero)
    V = omp.fit(np.matmul(np.transpose(U), A), G)

    res = list()
    length = len(V, 1)
    for i in range(1, length, step):
        left = np.floor(i)
        right = min(np.floor(i+block-1), length)
        if right < left:
            right = left

        v = np.split(V, [left, right+1], 1)
        if len(v, 1) == 1:
            v = np.hstack((v, v))
        mi = v.min(0)
        ma = v.max(0)
        s = ma
        for j in range(1, len(mi)):
            if -mi[j] > ma[j]:
                s[j] = mi[j]
        s = s/np.linalg.norm(s)
        res = np.hstack((res, np.transpose(s)))
    readData.dlmwrite(outpath, res)
    return res