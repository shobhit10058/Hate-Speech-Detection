import pandas as pd
import numpy as np
import os,string

# This function helps to read a TSV file
def ReadTsv(path):

    # input file 
    inp_f = open(path, 'r')

    # reading the columns and 
    # removing the next line character
    c = inp_f.readline()[:-1].split()

    # making a empty data frame with 
    # those columns
    frame = pd.DataFrame(columns=c)

    while(True):
        s = inp_f.readline()

        if(len(s) == 0):
            break

        l = s[:-1].split()

        # This dictionary will hold the contents 
        # of each column in a row
        d = {}

        # Following line just add the contents 
        # of a row to the data frame. These are 
        # made according to the training files 
        # and testing files we are using  
        if(len(c) == 3):
            d[c[0] ] = l[0]
            d[c[1] ] = " ".join(l[1:-1])
            d[c[2] ] = np.int64(l[-1])
            frame = frame.append(d, ignore_index=True)
        else:
            d[c[0] ] = l[0]
            d[c[1] ] = " ".join(l[1:])
            frame = frame.append(d, ignore_index=True)

    # Printing to a temporary file and reading it 
    # so that frame contains elements in proper format
    frame.to_csv('data/subs.csv', sep='\t', index=False)
    frame = pd.read_csv('data/subs.csv', sep='\t')
    os.remove('data/subs.csv')

    return frame

# The following function help remove the 
# punctuations and making lowercase the 
# text entries in our data 
def preprocess(data):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    i = 0
    for s in data['text']:
        s = s.translate(table)
        s = s.lower()
        data.at[i, 'text'] = s
        i += 1
