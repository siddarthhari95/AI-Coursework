import argparse
import math
parser = argparse.ArgumentParser()

 #first we use training file to prepare our training data and our naive bayes classifier model using the training data.
    ####################################global variables ##################
    
    
    #Have used logarithmic values to avoid floating point 

num_ham_words=0
num_spam_words=0
all_word_set = set()
num_mails = 0
c_p = {}
class_word_count={}
mail_type_count={}
dict_type_wordcount = {"ham":0,"spam":0}
#now we make a classifier which will update the conditionals for each type of  email word.

def do_math_classify(alpha,c,n,num):
    return float((c + alpha) / float(n + alpha*num))

def classifier(alpha):
    global num_mails
    global mail_type_count
    global c_p
    global class_word_count
    global training_f
    global dict_type_wordcount

    #we define a class_word_count dictionary which is a nested dictionary containing the counts of each word in each type of mail.
    class_word_count.setdefault("spam", {})
    class_word_count.setdefault("ham", {})
    # we start by making making a nested python dictionary that keeps a word count for each kind of 
    # mail, i.e respective word counts for all spam mails together and respective word counts for 
    # ham mails together.
    for w in all_word_set:
        class_word_count["spam"].setdefault(w, 0)
        class_word_count["ham"].setdefault(w, 0)
    training_f = open(training_file,'r')
    #print "before reading file"
    for line in training_f:
        #print "hello i am reading file"
        num_mails += 1
        line_elements = line.split(" ")
        type = line_elements[1]
        #count of mail of each type.
        mail_type_count.setdefault(type, 0)
        mail_type_count[type] += 1
        #print "hello",mail_type_count #################
        for index in range(2, len(line_elements), 2):
            class_word_count[type][line_elements[index]] += int(line_elements[index + 1])
    print "after reading file"
    
    #dict_type_wordcount is a dictionary which keeps a count of the total number of words 
    # for each type of mails taken together.
    
    for type in class_word_count:
        for w,c in class_word_count[type].items():
            dict_type_wordcount[type]+=c
            
    num_ham_words = dict_type_wordcount["ham"]
    num_spam_words = dict_type_wordcount["spam"]
    print mail_type_count
    for type in ["ham","spam"]:
        c_p.setdefault(type,{})
    
    n_s,n_h = dict_type_wordcount["spam"],dict_type_wordcount["ham"]
    
    num_unique_words=len(all_word_set)
    
    li1 = class_word_count.items()
    
    for class_t,dict in li1:
        li2 = dict.items()
        for wd,ct in li2:
            c_p[class_t].setdefault(wd, 0)
            if (class_t == "spam"):
                c_p[class_t][wd] = do_math_classify(alpha,ct,n_s,num_unique_words)  
            elif (class_t == "ham"):
                c_p[class_t][wd] = do_math_classify(alpha,ct,n_h,num_unique_words)

def do_math_test(a,b):
    return math.log10(1 / float( a+b))

def NB_prediction_on_test():
    global c_p
    global num_ham_words
    global num_spam_words
    global result_file
    global test_file
    global dict_type_wordcount
    
    result_file = open(result_file,'w')
    test_file = open(test_file,'r')
    
    true_label_li,pred_label_li = [],[]
    
    for line in test_file:
        line_elems= line.split()
        ID = line_elems[0]
        true_label = line_elems[1]
        wordss = [ w for i,w in enumerate(line_elems) if i%2==0]
        
        prob_h = 0.0
        prob_s = 0.0
        
        for word in wordss:
            if word not in all_word_set:
                prob_h = prob_h + do_math_test( dict_type_wordcount["ham"],len(all_word_set))
                prob_s = prob_s+ do_math_test(dict_type_wordcount["spam"],len(all_word_set))
            else:
                prob_h=prob_h + math.log10(c_p["ham"][word])
                prob_s=prob_s + math.log10(c_p["spam"][word])
            
                
        if prob_s > prob_h:
            pred_label = "spam"
        else:
            pred_label = "ham"    
    
        true_label_li+=[true_label]
        pred_label_li+=[pred_label]
        
        
        result_file.write(ID+" "+pred_label + "\n")
        
    c = 0
    for true,pred in zip(true_label_li,pred_label_li):
        if true == pred:
            print("hello")
            c+=1

    acc = c*0.1/(len(true_label_li) * 0.1)
    print "accuracy="+str(acc)
    #messageBox(true_label_li,pred_label_li)
        
    
if __name__ == "__main__":
    parser.add_argument('-f1',help = "give the training file",required = True)
    parser.add_argument('-f2',help = "give the test file making predictions", required = True)
    parser.add_argument('-o',help = "give the file where predictions for test will be stored", required = True)

    parsed_input= vars(parser.parse_args())

    training_file = parsed_input['f1']
    test_file = parsed_input['f2']
    result_file = parsed_input['o']
    training_f = open(training_file,'r')
    test_f = open(test_file,'r')
    #getting a set of all the unique words in the training vocabulary.
        
#so now we iterated through each line and got the number of total unique words in all_word_set
    for line in training_f:
        line_elements = line.strip("\n").split(" ")
        line_words_counts = line_elements[2:]
        line_words_li = [ word for i,word in enumerate(line_words_counts) if i%2==0]
        
        for w in line_words_li:    
            all_word_set.add(w)
            
            
    classifier(1)
    #various alpha values can be passed to classifier above.
    NB_prediction_on_test()
    

    
    
    
    
    
    
    