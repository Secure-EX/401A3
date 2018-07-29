import os
import re
import numpy as np
import time


dataDir = '/u/cs401/A3/data/'
# dataDir = "/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A3/data/"


def preprocess(s, steps=range(1, 15)):
    """ Pre-process the input list of string to raw human readable and calculatable
        For each line, preprocess these transcripts by removing all punctuation
        and setting the text to lowercase.

    :return: list of string
    """
    processing_str = []
    for word in s:
        word = word.lower()
        # replace any <.*>, [laughter], {bla bla}to ""
        if 1 in steps:
            word = re.sub(r"<.*>", "", word)
        # replace any [] to ""
        if 2 in steps:
            word = re.sub(r"\[.*\]", "", word)
        # replace any () to ""
        if 3 in steps:
            word = re.sub(r"\(.*\)", "", word)
        # replace any {} to ""
        if 4 in steps:
            word = re.sub(r"\{.*\}", "", word)
        # replace T/H:INTERACTIVE or LU/HE:SURVIVAL etc. to ""
        if 5 in steps:
            word = re.sub(r"\w+/\w+:\w+", "", word)
        # replace don't etc. to do n't
        if 6 in steps:
            word = re.sub(r"(\w+)(n't)", r"\1 \2", word)
        # replace I'll etc. to I 'll
        if 7 in steps:
            word = re.sub(r"(\w+)('\w*)", r"\1 \2", word)
        # replace h-, r- etc. stuff that will affect calculation
        if 8 in steps:
            word = re.sub(r"\w+-", "", word)
        # replace -t etc. stuff that will affect calculation
        if 9 in steps:
            word = re.sub(r"-\w+", "", word)
        # replace -t- etc. stuff that will affect calculation
        if 10 in steps:
            word = re.sub(r"-\w+-", "", word)
        # replace any punctuations to "", such as "-/", "./" the most greedy
        if 11 in steps:
            word = re.sub(r"[!\"#$%&()*+,._/:;<=>?@\[\]\^\{\|\}~-]", "", word)
        # finish modified the word, store into the processing_str list
        processing_str.append(word)
    # remove all items that are empty
    mod_str = list(filter(None, processing_str))
    return mod_str


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings
    h : list of strings
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    # >>> Levenshtein("who is there".split(), "is there".split())
    # 0.333 0 0 1
    # >>> Levenshtein("who is there".split(), "".split())
    # 1.0 0 0 3
    # >>> Levenshtein("".split(), "who is there".split())
    # Inf 0 3 0
    """
    # r is reference, n is the row numbers
    n = len(r)
    # h is hypothesis, m is the col numbers
    m = len(h)
    # Allocate matrix R[m+1, n+1], Matrix of distances
    R = np.full((n+1, m+1), np.inf)
    # 0th row should be initialized with [0, 1, 2, 3, 4, ...] and same goes for 0th column.
    # set [0, 1, 2, 3, 4, ... m] into first row
    R[0] = [i for i in range(m+1)]
    # set [0, 1, 2, 3, 4, ... n] into first col
    R[:, 0] = [i for i in range(n+1)]
    # Backtracking matrix
    B = np.full((n+1, m+1), np.inf)
    B[0].fill(-1)
    B[:, 0].fill(1)
    B[0, 0] = 0
    # start Levenshtein distance iteration
    for i in range(1, n+1):
        for j in range(1, m+1):
            del_cost = R[i-1, j] + 1
            sub_cost = R[i-1, j-1] + (r[i-1] != h[j-1])
            insert_cost = R[i, j-1] + 1
            R[i, j] = min(del_cost, sub_cost, insert_cost)
            # up
            if R[i, j] == del_cost:
                B[i, j] = 1
            # left
            elif R[i, j] == insert_cost:
                B[i, j] = -1
            # up-left
            else:  # R[i, j] == sub_cost
                B[i, j] = 2
    # print("R:")
    # print(R)
    # print("B:")
    # print(B)
    # count the number of the nS, nI and nD
    nS, nI, nD = 0, 0, 0
    i, j = n, m
    while i != 0 or j != 0:
        # up
        if B[i, j] == 1:
            nD += 1
            i, j = i - 1, j
        # left
        elif B[i, j] == -1:
            nI += 1
            i, j = i, j - 1
        # up-left
        else:
            nS += (r[i-1] != h[j-1])
            i, j = i - 1, j - 1
    WER = np.inf
    if n != 0:
        WER = 100 * R[n, m] / n
    return WER, nS, nI, nD


if __name__ == "__main__":
    start_time = time.time()
    # print(Levenshtein("who is there".split(), "is there".split()))
    # print(Levenshtein("who is there".split(), "".split()))
    # print(Levenshtein("".split(), "who is there".split()))
    # Format: [SPEAKER] [SYSTEM] [i] [WER] S:[numSubstitutions], I:[numInsertions], D:[numDeletions]
    spk = []
    sys = []
    ith = []
    wer_scores = []
    ns = []
    ni = []
    nd = []
    for subdir, dirs, files in os.walk(dataDir):
        for spe in dirs:
            # speaker name
            speaker = os.path.basename(spe)
            # file name and need to open simultaneously
            google = "transcripts.Google.txt"
            kaldi = "transcripts.Kaldi.txt"
            standard = "transcripts.txt"
            with open(dataDir + speaker + "/" + google, "r") as g,\
                 open(dataDir + speaker + "/" + standard, "r") as s:
                # compare standard and google
                for x, y in zip(s, g):
                    if len(x) == 0 or len(y) == 0:
                        continue
                    spk.append(speaker)
                    # system name
                    system = "Google"
                    sys.append(system)
                    x = preprocess(x.split())
                    y = preprocess(y.split())
                    # ith line
                    i = x[0]
                    # print("g" + i)
                    ith.append(i)
                    # WER, nS, nI, nD
                    levenshtein_sg = Levenshtein(x[1:], y[1:])
                    wer_scores.append(levenshtein_sg[0])
                    ns.append(levenshtein_sg[1])
                    ni.append(levenshtein_sg[2])
                    nd.append(levenshtein_sg[3])
            with open(dataDir + speaker + "/" + kaldi, "r") as k,\
                 open(dataDir + speaker + "/" + standard, "r") as s:
                # compare standard and kaldi
                for x, y in zip(s, k):
                    if len(x) == 0 or len(y) == 0:
                        continue
                    spk.append(speaker)
                    # system name
                    system = "Kaldi"
                    sys.append(system)
                    x = preprocess(x.split())
                    y = preprocess(y.split())
                    # ith line
                    i = x[0]
                    # print("k" + i)
                    ith.append(i)
                    # WER, nS, nI, nD
                    levenshtein_sk = Levenshtein(x[1:], y[1:])
                    wer_scores.append(levenshtein_sk[0])
                    ns.append(levenshtein_sk[1])
                    ni.append(levenshtein_sk[2])
                    nd.append(levenshtein_sk[3])
    index = 0
    kaldi_avg = []
    google_avg = []
    while index in range(len(spk)):
        if sys[index] == "Kaldi":
            kaldi_avg.append(wer_scores[index])
        else:
            google_avg.append(wer_scores[index])
        required_format = "{} {} {} {} S:{}, I:{}, D:{}"
        print(required_format.format(spk[index], sys[index], ith[index], wer_scores[index], ns[index], ni[index], nd[index]))
        index += 1
    # print(spk)
    # print(sys)
    # print(ith)
    # print(wer_scores)
    # print(ns)
    # print(ni)
    # print(nd)
    print("Kaldi avg:")
    print(np.mean(kaldi_avg))
    print("Google avg:")
    print(np.mean(google_avg))
    print("Kaldi std:")
    print(np.std(kaldi_avg))
    print("Google std:")
    print(np.std(google_avg))
    print("---Cost %s seconds to finish Levenshtein ---" % (time.time() - start_time))
