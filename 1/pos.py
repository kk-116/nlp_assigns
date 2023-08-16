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
    per_pos_accuracy = np.zeros((12,1))
    tot_pos = np.zeros((12,1))
    confusion_mat = np.zeros((14,14))
    tp = np.zeros((12,1))
    fp = np.zeros((12,1))
    fn = np.zeros ((12,1))
    prec = np.zeros((12,1))
    recall = np.zeros((12,1))
    f1_score = np.zeros((12,1))
    f2_score = np.zeros((12,1))
    fh_score = np.zeros((12,1))
    avg_p = 0.0
    tc = 0
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
            # print(word + "/" + str(index_tag(prev_tag)) + " ")
            tot_pos[tag_index(tag)-1] += 1
            confusion_mat[prev_tag][tag_index(tag)] += 1
            if prev_tag == tag_index(tag):
                per_pos_accuracy[prev_tag-1] += 1
                tp[prev_tag -1] += 1
            else:
                fn[prev_tag-1] +=1
                fp[tag_index(tag)-1] +=1
        avg_p += prob
        tc += 1
    ################### ADJUSTING ERROR ANALYSIS DATA FOR PRETTY PRINTING ###################
    avg_p = avg_p/tc
    print( "Average Sentence probability " + str(avg_p))
    per_pos_accuracy = per_pos_accuracy/tot_pos
    fp = fp/tot_pos
    fp1 = np.sum(fp)/np.sum(tot_pos)
    fn = fn/tot_pos
    fn1 = np.sum(fn)/np.sum(tot_pos)
    tp = tp/tot_pos
    tp1 = np.sum(tp)/np.sum(tot_pos)
    prec = tp/(tp + fp)
    prec1 = tp1/(tp1 +fp1)
    recall = tp/(tp + fn)
    recall1 = tp1/(tp1 + fn1)
    f1_score = 2/((1/prec)+(1/recall))
    f1_score1 = 2/((1/prec1)+(1/recall1))
    f2_score = 5*prec*recall/((4*prec)+recall)
    f2_score1 = 5*prec1*recall1/((4*prec1)+recall1)
    fh_score = 5*prec*recall/((4*recall)+prec)
    fh_score1 = 5*prec1*recall1/((4*recall1)+prec1)


    c_m = confusion_mat[1:13,1:13]
    c_m = c_m.T/c_m.sum(axis=1)
    c_m = c_m.T
    confusion_mat[1:13,1:13] = c_m

    return confusion_mat,per_pos_accuracy,prec,recall,f1_score,f2_score,fh_score,prec1,recall1,f1_score1,f2_score1,fh_score1

def train(data):
    # example of word_tag_freq
    # {"kamal":[0,0,0,1,0,22,3,4]}

    # universal tagset order :

    # VERB - verbs (all tenses and modes)
    # NOUN - nouns (common and proper)
    # PRON - pronouns 
    # ADJ - adjectives
    # ADV - adverbs
    # ADP - adpositions (prepositions and postpositions)
    # CONJ - conjunctions
    # DET - determiners
    # NUM - cardinal numbers
    # PRT - particles or other function words
    # X - other: foreign words, typos, abbreviations
    # . - punctuation


    # "word": [ VERB, NOUN, PRON, ADJ, ADV, ADJ, CONJ, DET, NUM, PRT, X, .]
    word_tag_freq = {}

    # calculating transition_matrix
    transition_matrix = np.zeros((14,14))
    # View of t_matrix
        # empty start tags.. end
        # start
        # tags
        # ...
        # end
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
    tm = transition_matrix[:13,1:]
    tm = tm.T/tm.sum(axis=1)
    tm = tm.T
    transition_matrix[:13,1:] = tm
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
    
    # for 5-fold CV
    data = np.array(data,dtype="object")
    data_ref = np.array_split(data,5)
    avg_f1_score = 0
    avg_fh_score = 0
    avg_f2_score = 0
    avg_prec = 0
    avg_recall = 0

    for i in range(5):
        print("Training/Testing Fold number: " + str(i+1))
        test = data_ref[i]
        fold = data_ref[:i] + data_ref[i+1:]
        # print(fold)
        train_data = []
        for t in fold:
            train_data.extend(t)
        
        ##################   Training our model   ##################
        transition_matrix,word_tag_freq = train(train_data)

        #####################   Testing and Collecting Error analysis data from model   #####################
        con_mat,per_pos_acc,prec,recall,f1_score,f2_score,fh_score,prec1,recall1,f1_score1,f2_score1,fh_score1 = hmm(test,transition_matrix,word_tag_freq)
        error_analysis = np.concatenate((prec,recall,fh_score,f1_score,f2_score),axis=1)
        
        avg_f1_score += f1_score1
        avg_fh_score += fh_score1
        avg_f2_score += f2_score1
        avg_prec += prec1
        avg_recall += recall1

 
        #################### PRETTY PRESENTATION OF ANALYSIS: ##############################
        # print(error_analysis)
        # print(per_pos_acc)
        # fig = go.Figure(data=[go.table(heade################ SCRAPE ################# SCRAP ############### SCRAPE ################### SCRAPE ######################## SCRAPE #############r=dict(values=['Precision','Recall','F1 Score']),cells = dict(values=[prec.tolist(),recall.tolist(),f1_score.tolist()]))])
        # fig.show()
        plot_pos_acc(per_pos_acc)
        plot_err(error_analysis)
        # plt.imshow(con_mat)
        # plt.pcolormesh(con_mat,cmap="coolwarm")
        ax = sb.heatmap(con_mat,linecolor="black",linewidth=0.5,cmap="coolwarm")
        plt.show()
        print("Fold " + str(i+1) + " training/testing done.")

    overall_analysis = np.array([avg_prec,avg_recall,avg_fh_score,avg_f1_score,avg_f2_score]).reshape((5,1))/5
    plot_over(overall_analysis)
    plt.show()
    ################ SCRAPE ################# SCRAP ############### SCRAPE ################### SCRAPE ######################## SCRAPE #############

    # # # getting tm and em
    # transition_matrix,word_tag_freq = train(data)
    # print(transition_matrix)
    # # # TODO: HMM implementation
    # test_input = ["While they are at the play , I am going to play with a dog ."]
    # test_data = split_test(test_input)
    # hmm(test_data, transition_matrix, word_tag_freq)
    # # # END TODO: HMM end

    ################ SCRAPE ################# SCRAP ############### SCRAPE ################### SCRAPE ######################## SCRAPE #############
    return 0

if __name__ == "__main__":
    main()