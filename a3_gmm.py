from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import numpy as np
import os
import fnmatch
import random
import time

# MFCCs were obtained using the python_speech_features module using default parameters,
# i.e., 25 ms windows, 13 cepstral coefficients, and 512 fast Fourier transform coefficients

# Each transcript file has the same format, where the ith line is: [i] [LABEL] [TRANSCRIPT]
# where ->
# [i] corresponds to i.wav and i.mfcc.npy,
# [LABEL] is the Global Lie label
# [TRANSCRIPT] is the actual transcript orthography.

# Global Lie valence and the version of the pre-interview task for the utterance appears
# before the colon (e.g., T/H) and the section name appears after the colon (e.g., INTERACTIVE).

# Global Lie valence is indicated as:
# T == Truth;
# LU == Lie Up (subject claims better performance than was actually achieved);
# LD == Lie Down (subject claims worse performance).

# Task version is indicated as:
# H == Hard;
# E == Easy.

# So, for example, T/H:INTERACTIVE indicates that the subject is telling the TRUTH based on
# having performed the HARD version of the INTERACTIVE task.

# npz files are all N rows and d-dimensional point in time (N x d)
# read .npz file: np.load("1.mfcc.npy")

# ! ignore any symbols that are not words !

dataDir = '/u/cs401/A3/data/'
# dataDir = "/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A3/data/"

class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        # omega shape => M x 1
        self.omega = np.zeros((M, 1))
        # mu shape => M x d
        self.mu = np.zeros((M, d))
        # Sigma shape => M x d
        self.Sigma = np.zeros((M, d))


def pre_compute_for_m(myTheta):
    """ helper
        precompute something for 'm' that applies to all x outside log_b_m_x(m, x, myTheta, preComputedForM=[]).
        Pass that precomputed component in preComputedForM[] in lob_b_m_x(m, x, myTheta, preComputedForM=[])
    """
    # sum M x d => a 1xM array since axis=1
    sub_equation_1 = -np.sum(np.divide(np.square(myTheta.mu), 2 * myTheta.Sigma), axis=1)
    # a real number
    sub_equation_2 = (-d/2) * np.log(2 * np.pi)
    # sum M x d => a 1xM array since axis=1
    sub_equation_3 = (-1/2) * np.sum(np.log(myTheta.Sigma), axis=1)
    # add them up we can get a 1xM array, include all M
    pre_compute = sub_equation_1 + sub_equation_2 + sub_equation_3
    # print("pre_compute_for_m computation done.")
    return pre_compute


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    """ Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside
        of this function.
        If you do this, you pass that precomputed component in preComputedForM
    """
    # x is a 1xd array, 1xd / 1xd and sum up a dxd matrix to a 1x1 real number at m position
    sub_equation_1 = - 1/2 * np.sum(np.divide(np.square(x), myTheta.Sigma[m]))
    # 1xd * 1xd / 1xd and sum up a dxd matrix to a 1x1 real number at m position
    sub_equation_2 = np.sum(np.divide(myTheta.mu[m] * x, myTheta.Sigma[m]))
    # 1x1 real number at m position + 1x1 real number at m position + M at m position return a 1x1 real number
    log_bmx = sub_equation_1 + sub_equation_2 + preComputedForM[m]
    # print("log_b_m_x computation done.")
    return log_bmx

    
def log_p_m_x(m, x, myTheta, preComputedForM=[]):
    """ Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
    See equation 2 of handout
    """
    log_bmx = log_b_m_x(m, x, myTheta, preComputedForM)
    # simplified the original function after natural log
    # log_pmx = real number + real number - real number
    log_pmx = np.log(myTheta.omega[m]) + log_bmx - logsumexp(log_bmx, b=myTheta.omega[m])
    # print("log_p_m_x computation done.")
    return log_pmx


def logLik(log_Bs, myTheta):
    """Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

    X can be training data, when used in train( ... ), and
    X can be testing data, when used in test( ... ).

    We don't actually pass X directly to the function because we instead pass:

    log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this
    function for efficiency.

    See equation 3 of the handout
    """
    # Mx1
    # use sum(logsumexp(log_Bs)) to get the log log_likelihood
    # omega x log_b => Mx1 x MxT => MxT
    # print(myTheta.omega.shape)
    # print(log_Bs.shape)
    # expect to be MxT
    # print(logsumexp(log_Bs, b=myTheta.omega))
    log_likelihood = np.sum(logsumexp(log_Bs, b=myTheta.omega))
    # print("logLikelihood computation done.")
    return log_likelihood


def compute_intermediate_result(X, T, myTheta):
    """ Create two MxT numPy arrays
        One to store each value from Equation 1 and the other to store each value from Equation 2.
    """
    preComputedForM = pre_compute_for_m(myTheta)
    # create log_Bs MxT matrix
    log_Bs = np.zeros((M, T))
    # create pmx MxT matrix
    pmx = np.zeros((M, T))
    for i in range(M):
        for j in range(T):
            log_Bs[i][j] = log_b_m_x(i, X[j], myTheta, preComputedForM)
            pmx[i][j] = np.exp(log_p_m_x(i, X[j], myTheta, preComputedForM))
    return log_Bs, pmx


def update_parameters(X, pmx):
    """ accomplished three equations in equation6
    """
    # X.shape[0] = T
    # shape pmx => MxT, shape X => Txd
    # np.sum(pmx, axis=1) == MxT => M =.T=> Mx1
    sum_pmx = np.transpose([np.sum(pmx, axis=1)])
    omega_hat = np.divide(sum_pmx, X.shape[0])
    # MxD
    mu_hat = np.divide(np.dot(pmx, X), sum_pmx)
    # MxD
    sigma_hat = np.divide(np.dot(pmx, np.square(X)), sum_pmx) - np.square(mu_hat)
    return omega_hat, mu_hat, sigma_hat


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    # Initialize theta
    T, d = X.shape[0], X.shape[1]
    myTheta = theta(speaker, M, X.shape[1])  # X is Txd, thus X.shape[0]=T, X.shape[1]=d
    # initialize omega randomly, with 0 < omega < 1 and SUM(omegas) = 1, try to set omegas to 1/M
    # myTheta.omega = np.zeros((M, 1))
    # for i in range(M):
    #     myTheta.omega[i] = 1/M
    # omega shape => M x 1
    myTheta.omega = np.transpose(np.random.dirichlet(np.ones(M), size=1))  # idea from stack overflow lol
    # initialize each mu to a random actual MFCC, which is the X, vector from the data
    # randomly draw M different MFCC vectors from X and then assign them to the M rows of myTheta.mu
    # mu shape => M x d
    myTheta.mu = np.zeros((M, d))
    for i in range(M):
        for j in range(d):
            myTheta.mu[i][j] = X[random.randrange(0, T)][j]
    # initialize Sigma_m to a identity matrix
    # Sigma shape => M x d
    myTheta.Sigma = np.ones((M, d))

    i = 0
    prev_L = - np.inf
    improvement = np.inf
    while i < maxIter and improvement >= epsilon:
        # ---------- TEST ---------- #
        # print("omega:")
        # print(myTheta.omega)
        # print(myTheta.omega.shape)
        # print("mu:")
        # print(myTheta.mu)
        # print(myTheta.mu.shape)
        # print("Sigma:")
        # print(myTheta.Sigma)
        # print(myTheta.Sigma.shape)
        # ---------- TEST ---------- #

        # ComputeIntermediateResults
        log_Bs = compute_intermediate_result(X, T, myTheta)[0]
        pmx = compute_intermediate_result(X, T, myTheta)[1]
        # ---------- TEST ---------- #
        # print("log_Bs:")
        # print(log_Bs)
        # print(log_Bs.shape)
        # print("pmx:")
        # print(pmx)
        # print(pmx.shape)
        # ---------- TEST ---------- #

        # ComputeLikelihood (X, myTheta)
        L = logLik(log_Bs, myTheta)

        # UpdateParameters (myTheta, X)
        myTheta.omega = update_parameters(X, pmx)[0]
        myTheta.mu = update_parameters(X, pmx)[1]
        myTheta.Sigma = update_parameters(X, pmx)[2]

        # improvement = Likelihood - previous_Likelihood
        improvement = L - prev_L
        # previous_Likelihood = Likelihood
        prev_L = L

        # ---------- TEST ---------- #
        # print(i)
        # print("log-likelihood:")
        # print(L)
        # print("improvment")
        # print(improvement)
        # ---------- TEST ---------- #

        # count iteration
        i += 1
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
        [ACTUAL_ID]
        [SNAME1] [LOGLIK1]
        [SNAME2] [LOGLIK2]
        ...
        [SNAMEK] [LOGLIKK]

        e.g.,
        S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    id_lst = []
    log_lik_lst = []
    MLE = - np.inf
    M = len(models[0].omega)
    T = len(mfcc)
    bestModel = -1
    for item in range(len(models)):
        # models[i] means the ith training model (theta model)
        preComputedForM = pre_compute_for_m(models[item])
        # compute log likelihood, use the format from compute_intermediate_result
        # create log_Bs MxT matrix
        log_Bs = np.zeros((M, T))
        for i in range(M):
            for j in range(T):
                # mfc is the X in train
                log_Bs[i][j] = log_b_m_x(i, mfcc[j], models[item], preComputedForM)
        L = logLik(log_Bs, models[item])
        id_lst.append(models[item].name)
        log_lik_lst.append(L)
        if L >= MLE:
            MLE = L
            bestModel = item

    # try to ouput the format!
    # print("correctID:")
    print(correctID)
    index = 0
    while index < k:
        required_format = "{} {}"
        print(required_format.format(id_lst[index], log_lik_lst[index]))
        index += 1
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    start_time = time.time()
    trainThetas = []
    testMFCCs = []
    # print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none, can change to 2 to test difference
    M = 8
    epsilon = 1.0
    maxIter = 3  # instead of 20, and i will use 10 as the max, and test 3 and 5 respectively
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
         numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect/len(testMFCCs)
    print("---Cost %s seconds to finish GMM ---" % (time.time() - start_time))
    print("Accuracy:")
    print(accuracy)
