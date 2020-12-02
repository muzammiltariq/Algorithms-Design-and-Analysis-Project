import time
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.interpolate import UnivariateSpline

'''                  NAIVE                        '''

#naive: O(m*(n-m+1) 
def naive(pat, txt):
    '''
    This function is the implementation of Naive Algorithm
    '''

    m = len(pat)
    n = len(txt)
    for i in range(n - m + 1):
        j = 0

        while(j < m):
            if (txt[i + j] != pat[j]):
                break
            j += 1

        if (j == m):
            pass

'''                  Knuth Morris                        '''

# KMP: O(m + n)
def Knuth_Morris(pat, txt):

    '''
    This function is the implementation of Knuth Morris Algorithm
    '''


    M = len(pat)
    N = len(txt)
    lps = [0]*M
    j = 0
    compute_LPS(pat, M, lps)

    i = 0
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            j = lps[j-1]
        elif i < N and pat[j] != txt[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1


def compute_LPS(pat, M, lps):

    '''
    This function is also an essential part of Knuth Mooris Implementation
    '''


    len = 0
    i = 1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            if len != 0:
                len = lps[len-1]
            else:
                lps[i] = 0
                i += 1

'''                  BOYER MOORE                        '''

#  BM: O(nm)
def bad_char_heuristic(string, size):

    '''
    This function is also an essential part of Boyer Moore
    '''

    badChar = [-1]*256
    for i in range(size):
        badChar[ord(string[i])] = i
    return badChar


def Boyer_Moore(pat, txt):
    '''
    This function is the implementation of Boyer Moore
    '''
    m = len(pat)
    n = len(txt)
    bad_char = bad_char_heuristic(pat, m)
    s = 0
    while(s <= n-m):
        j = m-1
        while j >= 0 and pat[j] == txt[s+j]:
            j -= 1
        if j < 0:
            s += (m-bad_char[ord(txt[s+m])] if s+m < n else 1)
        else:
            s += max(1, j-bad_char[ord(txt[s+j])])





'''                  SIMULATION                        '''

def wordgenerator(n):
    '''
    This function is used to generate strings of length n randomly.
    
    input: n -> length of the string to be created
    Output: txt -> a randomly generated string  
    
    '''
    choice = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # choice = "10" 
    txt = [random.choice(choice) for _ in range(n)]
    txt = ''.join(txt)

    return txt

def complexity_n(pat,txt,algo):
    '''
    This function is used to return a list of time taken by each algorithm to perform string matching
    keeping pattern constant while increasing the size of text with every iternation

    input: pattern, text, algorithm (to be used)
    output: time_taken -> List of time taken by the algorithm 
    '''
    time_taken = []
    n = len(txt)
    m = len(pat)
    # pat = wordgenerator(m)
    for i in range(m,n):
        start = time.time()
        algo(pat,txt[0:i]) #keeping pat constant while increasing the size of the txt with every iteration
        time_taken.append(time.time() - start)
    
    return time_taken

def complexity_m(txt,pat,algo):
    '''
    This function is used to return a list of time taken by each algorithm to perform string matching
    keeping text constant while increasing the size of pattern with every iternation

    input: pattern, text, algorithm (to be used)
    output: time_taken -> List of time taken by the algorithm 
    '''

    time_taken = []
    n = len(txt)
    m = len(pat)
    # pat = wordgenerator(m)
    for i in range(m):
        start = time.time()
        algo(pat[0:i+1],txt) # keeping txt constant while increasing the size of the pattern in each iteration
        # algo(pat,txt)
        time_taken.append(time.time() - start)

    return time_taken

def n_procedure():

    '''
    
    This function is used to plot the complexities of the Algorithms
    
    '''

    n = 10000
    m = 500

    txt = wordgenerator(n)
    pat = wordgenerator(m)
    
    # KMP Simulation
    time_taken_n = np.array(complexity_n(pat,txt,Knuth_Morris)) 
    x = np.array(list(range(m,len(time_taken_n)+m)))

    s = UnivariateSpline(x, time_taken_n, s=5)
    xs = np.linspace(0, n, n)
    ys = s(xs)
    plt.plot(xs, ys, "-r", label = "Knuth Morris")

    print("KMP Completed")


    # NB Simulation
    time_taken_n = np.array(complexity_n(pat,txt,naive)) 
    s = UnivariateSpline(x, time_taken_n, s=5)
    n_xs = np.linspace(0, n, n)
    n_ys = s(n_xs)
    plt.plot(n_xs,n_ys, "-b",label = "Naive")

    # Boyer Moore Simulation
    time_taken_n = np.array(complexity_n(pat,txt,Boyer_Moore)) 
    s = UnivariateSpline(x, time_taken_n, s=5)
    b_xs = np.linspace(0, n, n)
    b_ys = s(b_xs)
    plt.plot(b_xs,b_ys, "-g", label = "Boyer Moore")


    plt.legend(loc="upper left")
    plt.xlabel("Length of text(n)")
    plt.ylabel("Time Taken")

    plt.show()

def m_procedure():

    '''
    
    This function is used to plot the complexities of the Algorithms
    
    '''

    n = 10000
    m = 10000   

    txt = wordgenerator(n)
    pat = wordgenerator(m)

    # KMP Simulation
    time_taken_m = np.array(complexity_m(pat,txt,Knuth_Morris))
    x = np.array(list(range(len(time_taken_m))))

    s = UnivariateSpline(x, time_taken_m, s=5)
    xs = np.linspace(0, m, m)
    ys = s(xs)
    plt.plot(xs, ys, "-r", label = "Knuth Morris")

    print("KMP Completed")


    # NB Simulation
    time_taken_m = np.array(complexity_m(pat,txt,naive))
    s = UnivariateSpline(x, time_taken_m, s=5)
    n_xs = np.linspace(0, m, m)
    n_ys = s(n_xs)
    plt.plot(n_xs,n_ys, "-b",label = "Naive")
    print("Naive Completed")


    # Boyer Moore Simulation
    time_taken_m = np.array(complexity_m(pat,txt,naive))
    s = UnivariateSpline(x, time_taken_m, s=5)
    b_xs = np.linspace(0, m, m)
    b_ys = s(b_xs)
    plt.plot(b_xs,b_ys, "-g", label = "Boyer Moore")
    print("Boyer Moore Completed")

    plt.legend(loc="upper left")
    plt.xlabel("Length of Pattern(m)")
    plt.ylabel("Time Taken")

    plt.show()


# n_procedure()
m_procedure()

