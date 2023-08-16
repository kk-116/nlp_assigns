from ast import main
from ctypes.wintypes import tagSIZE
import json
from tracemalloc import start
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sb

from nltk.corpus import brown

def plot_pos_acc(per_pos_acc):
    plt.rcParams["figure.figsize"] = [12.0, 3.0]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    columns = ("Accuracy")
    rows = ("VERB","NOUN","PRON","ADJ","ADV","ADP","CONJ","DET","NUM","PRT","X",".")
    axs.axis('tight')
    axs.axis('off')
    axs.set_title("Per Pos Accuracy")
    the_table = axs.table(cellText=per_pos_acc, colLabels=columns,rowLabels=rows, loc='center')
    # plt.savefig("error_analysis")
    plt.show()
    pass


def plot_over(overall_analysis):
    plt.rcParams["figure.figsize"] = [10.0, 2.0]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    rows = ("Precision", "Recall", "F0.5_Score","F1_Score","F2_Score")
    columns = ("Overall")
    axs.axis('tight')
    axs.axis('off')
    axs.set_title("Error Analysis")
    the_table = axs.table(cellText=overall_analysis, colLabels=columns,rowLabels=rows, loc='center')
    # plt.savefig("error_analysis")
    plt.show()

def plot_err(error_analysis):
    plt.rcParams["figure.figsize"] = [12.0, 5.0]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 1)
    columns = ("Precision", "Recall", "F0.5_Score","F1_Score","F2_Score")
    rows = ("VERB","NOUN","PRON","ADJ","ADV","ADP","CONJ","DET","NUM","PRT","X",".")
    axs.axis('tight')
    axs.axis('off')
    axs.set_title("Error Analysis")
    the_table = axs.table(cellText=error_analysis, colLabels=columns,rowLabels=rows, loc='center')
    # plt.savefig("error_analysis")
    plt.show()

def split_test(test_data):
    list = []
    for i in test_data:
        list1 = []
        for j in i.split():
            list1.append(((j,"a")))
        list.append(list1)
            
    return list


def index_tag(tag):
    if tag == 0:
        return "START"
    elif tag == 1:
        return "VERB"
    elif tag == 2:
        return "NOUN"
    elif tag == 3:
        return "PRON"
    elif tag == 4:
        return "ADJ"
    elif tag == 5:
        return "ADV"
    elif tag == 6:
        return "ADP"
    elif tag == 7:
        return "CONJ"
    elif tag == 8:
        return "DET"
    elif tag == 9:
        return "NUM"
    elif tag == 10:
        return "PRT"
    elif tag == 11:
        return "X"
    elif tag == 12:
        return "."
    elif tag == 13:
        return "END"
    else:
        print("wrong tag index\n")

def tag_index(tag):
    if tag == "START":
        return 0
    elif tag == "VERB":
        return 1
    elif tag == "NOUN":
        return 2
    elif tag == "PRON":
        return 3
    elif tag == "ADJ":
        return 4
    elif tag == "ADV":
        return 5
    elif tag == "ADP":
        return 6
    elif tag == "CONJ":
        return 7
    elif tag == "DET":
        return 8
    elif tag == "NUM":
        return 9
    elif tag == "PRT":
        return 10
    elif tag == "X":
        return 11
    elif tag == ".":
        return 12
    elif tag == "END":
        return 13
    else:
        print("wrong tag\n")

# Hidden Markov Model
def hmm(test_data, tm, em):
   
    for sent in test_data:
        prev_tag = 0
        prob = 1.0
        for word,tag in sent:
            if word.lower() in em.keys():
                emXtm = np.array(em[word.lower()])*tm[prev_tag]
                max_prob = np.max(emXtm)
                prev_tag = np.where(emXtm == max_prob)[0][0]
                prob *= max_prob
            else:
                # limiting to open class pos tags (noun,verb,adjective and adverb) to solve unknown words
                
                max_prob = np.max(np.concatenate((tm[:,1:3],tm[:,4:5]),axis=1)[prev_tag])
                prev_tag = np.where(tm[prev_tag] == max_prob)[0][0]
                prob *= max_prob
            print(word + "/" + str(index_tag(prev_tag)) + " ")
          
           
def train(data):
    word_tag_freq = {}

    # calculating transition_matrix
    transition_matrix = np.zeros((14,14))

    for sent in data:
        prev_tag = "START"
        for word,tag in sent:
            transition_matrix[tag_index(tag)][tag_index(prev_tag)] += 1
            prev_tag = tag
            if word.lower() in word_tag_freq.keys():
                word_tag_freq[word.lower()][tag_index(tag)] += 1
                pass
            else:
                word_tag_freq[word.lower()] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                word_tag_freq[word.lower()][tag_index(tag)] = 1
                pass
        transition_matrix[tag_index("END")][tag_index(prev_tag)] += 1
    transition_matrix = transition_matrix.T
    np.savetxt('test.out', transition_matrix, delimiter=',') 
    tm = transition_matrix[:13,1:]
    np.savetxt('test1.out', tm, delimiter=',') 
    tm = tm/tm.sum(axis=1)
    # tm = tm.T
    transition_matrix[:13,1:] = tm
    np.savetxt('test2.out', transition_matrix, delimiter=',')   # X is an arra
    # print(tm)
    # print(transition_matrix)
    # we are done calculating required tables by here
    em = {}
    for key in word_tag_freq.keys():
        a = np.sum(np.array(word_tag_freq[key]))
        em[key] = np.array(word_tag_freq[key])/a
    return transition_matrix,em

def main():
    # for final submission 
    data = list(brown.tagged_sents(tagset='universal'))

    # f = open("./data.json",'r')
    # data = json.load(f)
    # f.close()
     
   
    
    ################ SCRAPE ################# SCRAP ############### SCRAPE ################### SCRAPE ######################## SCRAPE #############

    # # # getting tm and em
    transition_matrix,word_tag_freq = train(data)
    # print(transition_matrix)
    # # # TODO: HMM implementation
    test_input = ["While they are at the play , I am going to play with a dog ." ,"I want to play the play ."]

    test_data = split_test(test_input)
    hmm(test_data, transition_matrix, word_tag_freq)
    # # # END TODO: HMM end

    ################ SCRAPE ################# SCRAP ############### SCRAPE ################### SCRAPE ######################## SCRAPE #############
    return 0

if __name__ == "__main__":
    main()