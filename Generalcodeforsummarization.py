# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:32:59 2018

@author: Shyamal.Banerjee
"""


import os
os.chdir('C:\\Users\\shyamal.banerjee\\Desktop\\NLP-summarisation problem\\Call transcripts and other company reports\\All new company transcripts')
import nltk
import string
import pandas as pd
import numpy as np
import math
import re
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
lemmer = nltk.stem.WordNetLemmatizer()
from nltk.wsd import lesk

##Function for converting string to lowercase and removal of punctuations##
def remove_punctuation(text):
        words = nltk.word_tokenize(text)
        pun_removed = [w for w in words if w.lower() not in string.punctuation]
        return " ".join(pun_removed) 

##Function for removal of tokens with updation of token indices##
def remove_tokens(tokens, ind):
    l = len(ind)    
    for i in range(l):
        del tokens[ind[0]]
        del ind[0]
        for j,k in enumerate(ind):
            ind[j] = ind[j]-1
    return tokens

##Function for replacement of newline characters and other unimportant characters from dictionary of characters##
def replace_char(x):
    rep_dict = {"\\n":" ", "\\xoc":" ", "\n":" ", "\r":" ", "\x0c":" ", "Print Page Close Window":""}
    for r in rep_dict:
        x = x.replace(r, rep_dict[r])
    return x

##Function for appending sections of a document##
def append_to_doc(tokens1, tokens2):
    for t in tokens2:
        tokens1.append(t)
    return tokens1

##replaces dictionary keys with values in a document##
def replace_in_doc(file_content):
    rep_dict = {"U.S.":"US","U.K.":"UK","U.A.E.":"UAE","Govt.":"Govt","Tier 3":"Tier III",
                "first half":"H1","second half":"H2",",000 barrels per day":"kb/d",
                "avg.":"average","approx.":"approximately", "incl.":"including"}
    for r in rep_dict:
        file_content = file_content.replace(r, rep_dict[r])
    return file_content

##Function for removal of blanks and storing their token indices##
def remove_blanks(para):
    ind = []
    for i,p in enumerate(para):
        if p == "":
            ind.append(i)
    return(remove_tokens(para, ind))

##Function for removal of tokens with unimportant keywords and phrases and appending of sections##
def remove_unimportant(para):
    unimportant=["dividend","dividends","Dividend","payout","cash returns","net income","adjusted net income",
                 "share repurchase","undervalued","overvalued","amortization"," depreciation","reconciliation",
                 "debt","capital structure","market cap","Q&A","Operator","earning","earnings",
                 "debt-to-capitalization","financial position","liquidity","debt-to-cap","Thank you",
                 "Thanks","Good morning","thanks","Investor Relations","closing remarks","opening remarks",
                 "administrative expenses","conference call","disconnect","share repurchases","congrats",
                 "congratulations","Congrats","Congratulations","pay-out","working capital","receivables",
                 "buying back shares","beat a dead horse","SEC"]
    for i,p in enumerate(para):
        sent = sent_tokenize(p)
        ind = []
        for j,s in enumerate(sent):
            s  = s.lower()
            t_dummy = word_tokenize(s)
            t_dummy_1=get_ngrams(s,2)
            t_dummy_2=get_ngrams(s,3)
            t_dummy_3=get_ngrams(s,4) 
            for word in unimportant:
                if  word in t_dummy:
                    ind.append(j)
                    break
                elif word in t_dummy_1:
                    ind.append(j)
                    break
                elif word in t_dummy_2:
                    ind.append(j)
                    break
                elif word in t_dummy_3:
                    ind.append(j)
                    break
        temp = ""
        for j in range(len(sent)):
            if j not in ind:
                temp = temp + " " + sent[j]
            para[i] = temp
    return(remove_blanks(para))
            



#os.chdir('C:\\\\Users\\\\Debraj.Bose\\\\Documents\\\\Project\\\\Data')
##reading files from working directory##
all_files = os.listdir("C:\\Users\\shyamal.banerjee\\Desktop\\NLP-summarisation problem\\Call transcripts and other company reports\\All new company transcripts")
##Filtering out the text files##
filenames = list(filter(lambda x: x[-4:] == '.txt', all_files))

# =============================================================================
# filenames=["VLO_2017_Q4_Combined.txt","PSX_2017_Q4_Combined.txt","MPC_2017_Q4_combined.txt",
#            "BP_2017_Q4_Combined.txt","BP_2018_Q1_Combined.txt","CVX_2017_Q4_Combined.txt",
#            "CVX_2018_Q1_Combined.txt",
#            "PSX_2018_Q1_Combined.txt","RDS_2017_Q4_Combined.txt","RDS_2018_Q1_Combined.txt"]
# =============================================================================
for k,t in enumerate(filenames):
    print(k,t)
yourvar = input('Choose a file index: ')
print('you entered: ' + yourvar)

file_content = open(filenames[int(yourvar)]).read()
file_content = replace_in_doc(file_content)

##Function for removal of short sentences with <6 words##
def remove_less_than_six(para):
    for i,m in enumerate(para):
        sent = sent_tokenize(m)
        ind = []
        for j,s in enumerate(sent):
            t_dummy=remove_punctuation(s)
            if len(word_tokenize(t_dummy))<6:
                ind.append(j)
        temp = ""
        for j in range(len(sent)):
            if j not in ind:
                temp = temp + " " + sent[j]
            para[i] = temp
    return para

##Creating dictionary of all paragraph indices##
test = dict()
for match in re.finditer(r'(?s)((?:[^\n][\n]?)+)', file_content):
   #print (match.start(), match.end())
    test[match.end()] = match.start() 
    
##Deleting paragraph just before Q&A##
#for m in re.finditer(r'\n\nQ&A',file_content) 
# =============================================================================
# for m in re.finditer(r'Q&A \n\n',file_content):
#     a = m.start()
# 
# b = test[a+1]
# file_content[b:a]
# file_content=file_content.replace(file_content[b:a],"")
# =============================================================================

test = dict()
for match in re.finditer(r'(?s)((?:[^\n][\n]?)+)', file_content):
   #print (match.start(), match.end())
    test[match.end()] = match.start() 
    
##Extracting all the paragraphs in a list before tokenizing##
a=list()
for key,value in test.items():
    a.append(file_content[value:(key+1)])
 
##Extracting speaker names in an array##    
#words=nltk.word_tokenize(str(a[2:6]))
#e=words[2:56]
for k,t in enumerate(a):
    if t.startswith("Company Participants \n\n"):
        startIndex1=k
        print(startIndex1)
        while not a[k].endswith("\nMANAGEMENT DISCUSSION SECTION \n\n"):
            endIndex=k+1
            k=k+1
        print(endIndex)
        break

st=' '.join(a[startIndex1:(endIndex+1)])
words=nltk.word_tokenize(st)

for k,t in enumerate(words):
    if "•" in t:
        startIndex=k+1
        print(startIndex)
        while not ("MANAGEMENT" in words[k+1]):
            endIndex=k+1
            k=k+1
        break
        print(endIndex)

e=words[startIndex:(endIndex+1)]
# =============================================================================
# ##Sentence tokenizing the Press releases##
# st=' '.join((a[0:startIndex1]))
# PRtokens=nltk.sent_tokenize(st)
# #print(PRtokens)
# #Tagslist in PR for deleting the entire section below the tags#
# tagslist=["About Valero","Conference Call","Investor Webcast","NOTICE",
#           "NOTES TO THE UNAUDITED"]
# for k,t in enumerate(PRtokens):
#      for tag in tagslist:
#         if tag in t:
# #    if "About Valero" in t:
#          start=k
#         #print(startIndex)
# endIndex=len(PRtokens)
# del PRtokens[start:endIndex]
# #print(PRtokens)   
# 
# ##Removing all tokens with inverted commas##
# start_pt=[0]*len(PRtokens)
# end_pt=[0]*len(PRtokens)
# ind=[]
# for k,t in enumerate(PRtokens):
#     start_pt[k] = t.find("“")
#     end_pt[k] = t.find("”", start_pt[k] + 1)  # add one to skip the opening "
#     quote = PRtokens[k][start_pt[k] + 1: end_pt[k] + 1]  # add one to get the quote excluding the ""
#     if end_pt[k]!=-1:
#         ind.append(k)
# len(PRtokens)        
# PRtokens=remove_tokens(PRtokens,ind)
# l1=len(PRtokens)
# =============================================================================

##Deleting from beginning of ECT upto start of speech for first speaker##
# =============================================================================
# del a[1:6]
# =============================================================================

for k,t in enumerate(a):
    if t.startswith("Company Participants \n\n"):
        startIndex=k
        print(startIndex)
        while not a[k+1].startswith(e[0]):
            endIndex=k+1
            k=k+1
        print(endIndex)
        break
del a[startIndex:(endIndex+1)]

##Appending all the Q&A chunks post Q&A##
qna=[]
startIndex=[]
ctr = 0
for k,t in enumerate(a):
    if t.startswith("<Q") or t.startswith("\x0c\n<Q") or t.startswith(" \n<Q"):
        startIndex.append(k)
        #print(startIndex)
        while not(a[k+1].startswith("<Q") or a[k+1].startswith("Operator \n\n") or
                  a[k+1].startswith("\x0c\nOperator \n\n") or
                  a[k+1].startswith("\x0c\n<Q") or
                  a[k+1].startswith(e[0]) or
                  a[k+1].startswith(" \n<Q")):
            k=k+1
        endIndex=k+1
        #print(endIndex)
        temp = ""
        for i in range(startIndex[ctr], endIndex):
            temp = temp + " " + a[i]
        qna.append(temp)
        ctr += 1

##PR+pre Q&A + Q&A chunks##


#tokens=append_to_doc(PRtokens,a[1:(startIndex[0])])
tokens=append_to_doc(a[0:startIndex[0]],qna)
len(tokens)

##Function for extraction of n-grams##
from nltk.util import ngrams
from collections import Counter
        
def get_ngrams(tokens, n):
    n_grams = ngrams(nltk.word_tokenize(tokens),n)
    return [' '.join(grams) for grams in n_grams] 

##Extracting the first speaker portion##
for k,t in enumerate(tokens):
    if t.startswith(e[0]):
        startIndex=k
        print(startIndex)
        while not(tokens[k].startswith("Q&A \n\n") or tokens[k].startswith(e[4]) or 
                  tokens[k].startswith(e[3])):
            endIndex=k+1
            k=k+1
        print(endIndex)
        break

##Running a loop over the first speaker portion and deleting the first speaker portion from
# the beginning to where the word in the wordslist is encountered##
wordslist=["SEC","Safe Harbor","Stock Exchange Announcement","Forward-looking statements",
           "cautionary statement","slide 2","with me today",
           "not received the earnings release and would like a copy"]
for i in range((endIndex-1),(startIndex-1),-1):
    t_dummy=nltk.word_tokenize(tokens[i])
    t_dummy_1=get_ngrams(tokens[i],2)
    t_dummy_2=get_ngrams(tokens[i],3)
    t_dummy_10=get_ngrams(tokens[i],10)
    flag = 0
    for word in wordslist:
        if word in t_dummy:
            flag = 1
            break
        elif word in t_dummy_1:
            flag = 1
            break
        elif word in t_dummy_2:
            flag = 1
            break
        elif word in t_dummy_10:
            flag = 1
            break
    if flag == 1:
        del tokens[startIndex: (i+1)]
        break

##Replacing speaker names with blanks##
for k,t in enumerate(tokens):
    for speaker_name in e:
        if speaker_name in t:
            #print (k)
            #print ('BEFORE')
            #print (tokens[k])
            tokens[k]=tokens[k].replace(speaker_name,'')


##Apending indices and tokens where the list of keywords or combination of keywords and 
# phrases appear##
capex=[]
indcapex=[]
capex_wordslist=["Project underway","Unit","Catalytic reforming",
           "Ethane cracker","Construction","Start-up","Complet",
           "Commissioning","Project","Acquisition","Merger","Plant","Expansion","added","addition","adding",
           "Moderniz","Investment","Upgrade","Contract","PSC",
           "Mechanical completion","Phase","Extension agreement","Partnered",
           "projects","capital investment plans","capex",
           "Project underway","capacity expansion","Capacity utilization","invest",
           "acquisition of assets","year plans","capital plan","growth capital","capital allocation",
           "Capital expenditure","set to deliver","capital spending","spending capital","capital program",
           "modernization","capex","completion","install","committed capital","capital investment","capital investments",
           "construct","expand","expansion","Final Investment decision (FID)","capacity",
           "start up","starting up","Plans to spend","spent on","Hurdle rate","to be spent on","CapEx",
           "Capital expenditure","capex","discovery","Brownfield","greenfield","Divestment","asset sales","sold stake",
           "process of selling","stake","asset sale","replac","selling assets","sell assets"
           "Bought","acquired","acquisition",
           "merged","merger","under evaluation","upgrade","upgradation","signed agreements","concession contract",
           "brought on-stream","discovery","will be establishing","license"]

# =============================================================================
# name = {}
# for w in capex_wordslist:
#     name[w]=0
# =============================================================================
    
for i in range(0,(len(tokens))):
# =============================================================================
#     if i in marketind:
#         continue
# =============================================================================
    t = (tokens[i]).lower()
    t_dummy=nltk.word_tokenize(t)
    t_dummy_1=get_ngrams(t,2)
    t_dummy_2=get_ngrams(t,3)
    t_dummy_3=get_ngrams(t,4)
    t_dummy_4=get_ngrams(t,5)
    flag = 0
    for word in capex_wordslist:
        if word.lower() in t_dummy:
            flag = 1
            break
        elif word.lower() in t_dummy_1:
            flag = 1
            break
        elif word.lower() in t_dummy_2:
            flag = 1
            break
        elif word.lower() in t_dummy_3:
            flag = 1
            break
        elif word.lower() in t_dummy_4:
            flag = 1
            break
    if flag == 1:
        capex.append(tokens[i])
        indcapex.append(i)
# =============================================================================
#         name[word] += 1
# =============================================================================
    elif ("signing" in t or "sign" in t) and ("contracts" in t or "contract" in t):
        capex.append(tokens[i])
        indcapex.append(i)
    elif ("divestment" in t or "asset sales" in t or "solid stake" in t or "process of selling" in t) and ("asset sale" in t or "replaced" in t or "replacement" in t):
        capex.append(tokens[i])
        indcapex.append(i)
    elif "sale" in t and ("distribution and marketing rights" in t or "retail sites" in t or "interest" in t):
        capex.append(tokens[i])
        indcapex.append(i)
    elif ("upgrade" in t or "upgradation" in t) and ("refinery" in t or "plant" in t):
        capex.append(tokens[i])
        indcapex.append(i)
    elif "grow" in t and "spend on" in t:
        capex.append(tokens[i])
        indcapex.append(i)
    elif "add" in t and "upstream" in t and "production" in t:
        capex.append(tokens[i])
        indcapex.append(i)
    elif "brought" in t and "on-stream" in t:
        capex.append(tokens[i])
        indcapex.append(i)
    elif ("plan" in t or "design" in t) and ("to produce" in t or "to launch" in t):
        capex.append(tokens[i])
        indcapex.append(i)
    elif "expect" in t and "to develop" in t:
        capex.append(tokens[i])
        indcapex.append(i)
    elif "acquire" in t and "interest" in t:
        capex.append(tokens[i])
        indcapex.append(i)
    elif ('pipeline' in t or 'terminal' in t or "unit" in t or "alkylation" in t or "isomerization" in t):
        tagged = nltk.pos_tag(word_tokenize(tokens[i]))
        for w,tag in tagged:
            if tag == 'NNP' or tag == "NNPS":
                capex.append(tokens[i])
                indcapex.append(i)
                break
            
#print(capex)
print(indcapex)
len(capex)

capex = remove_less_than_six(capex)
capex = remove_unimportant(capex)
for i,m in enumerate(capex):
    capex[i]=replace_char(m)

##Appending indices and list of tokens where the list of keywords and combination of 
#keywords appear for marketoutlook appear##
marketoutlook=[] 
marketind=[]
market_wordslist=["EPA","RIN","RON","CAFE","regulat","policy","sanction",
           "IMO","Tier 3","RFS","PDS","compliance",
           "geopolitical","Macro","Market impact","market outlook",
           "market implications","geopolitic",
           "macroeconomic","Electric vehicles","inflation","WTI-WCF",
           "OPEC","OECD",
           "landscape",
           "Pacific basin","Atlantic basin","oversupplied","undersupplied",
           "perception","structurally",
           "renewable","oil sands","election",
           "estimate","forecast","expect","predict","short term","medium term",
           "long term","opportunit","Permian","Cushing",
           "Brent-WTI","WTI Brent","differential"]

future = ["will","would","plan","may","might","could","can","anticipate","forward-looking"]

#confusion = ["Brent-WTI", "WTI Brent", "WTI-WCF", "spreads", "benchmark", "Brent-Dubai",]

for i in range(0,(len(tokens))):
    if i in indcapex:
        continue
    s = (tokens[i]).lower()
    t_dummy=nltk.word_tokenize(s)
    t_dummy_1=get_ngrams(s,2)
    t_dummy_2=get_ngrams(s,3)
    t_dummy_3=get_ngrams(s,4)
    t_dummy_4=get_ngrams(s,5)
    flag = 0
    for word in market_wordslist:
        if word.lower() in t_dummy:
            flag = 1
            break
        elif word.lower() in t_dummy_1:
            flag = 1
            break
        elif word.lower() in t_dummy_2:
            flag = 1
            break
        elif word.lower() in t_dummy_3:
            flag = 1
            break
        elif word.lower() in t_dummy_4:
            flag = 1
            break
    if flag == 1:
        marketoutlook.append(tokens[i])
        marketind.append(i)
    elif ("global" in s or "worldwide" in s) and ("demand" in s or "supply" in s or "maintenance" in s or "economic growth" in s):
        marketoutlook.append(tokens[i])
        marketind.append(i)
    elif ("shift" in s or "switching" in s) and "natural gas" in s:
        marketoutlook.append(tokens[i])
        marketind.append(i)
    elif "renewable" in s and "grow" in s:
        marketoutlook.append(tokens[i])
        marketind.append(i)
    elif ("demand" in t or "supply" in t or "industry" in t) and "driver" in t:
        marketoutlook.append(tokens[i])
        marketind.append(i)
    elif ("economy" in t or "environment" in t) and ("demand" in t or "supply" in t):
        marketoutlook.append(tokens[i])
        marketind.append(i)
    
        
#print(marketoutlook)
print(marketind)
len(marketoutlook)

marketoutlook = remove_less_than_six(marketoutlook)
marketoutlook = remove_unimportant(marketoutlook)
for i,m in enumerate(marketoutlook):
    marketoutlook[i]=replace_char(m)

##Appending indices and list of tokens where keywords and combination of keywords and 
#phrases from Operational Highlights appear##
##Operational Highlights Section##
operational=[]
indoperational=[]
companylist=["Valero","VLO","Phillips","PSX","Marathon","MPC","Chevron","CVX","Shell","RDS","BP"]
operational_wordslist=["slate","feedstock","logistic","Valero","VLO","Phillips","PSX","Marathon","MPC",
                       "Chevron","CVX","Shell","RDS","BP","operation","operating","cycle",
                       "margin","CY 2017","current year 2017","shutdown","market share","quarter",
                       "hedg","start","trade","trading","consistent","non-producing asset",
                       "asset impairment","risk reward ratio","reserve replacement ratio","compare",
                       "relying on","cash balanced","IPO","profitability","performance","marketing",
                       "retail","turnaround","coking","distillate","gasoline","residual","chemicals",
                       "OPEX"]

for i in range(0,(len(tokens))):
    if i in marketind or i in indcapex:
        continue
    t = (tokens[i]).lower()
    t_dummy=nltk.word_tokenize(t)
    t_dummy_1=get_ngrams(t,2)
    t_dummy_2=get_ngrams(t,3)
    t_dummy_3=get_ngrams(t,4)
    t_dummy_4=get_ngrams(t,5)
    flag = 0
    for word in operational_wordslist:
        if word.lower() in t_dummy:
            flag = 1
            break
        elif word.lower() in t_dummy_1:
            flag = 1
            break
        elif word.lower() in t_dummy_2:
            flag = 1
            break
        elif word.lower() in t_dummy_3:
            flag = 1
            break
        elif word.lower() in t_dummy_4:
            flag = 1
            break
    if flag == 1:
        operational.append(tokens[i])
        indoperational.append(i)
    elif ("processing" in t or "processed" in t or "proportion" in t) and "crude" in t and "%" in t:
        operational.append(tokens[i])
        indoperational.append(i)
    elif ("Q1" in t or "Q3" in t) and ("Q2" in t or "Q4" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif "2017" in t and ("2016" in t or "2018" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif ("export" in t or "import" in t or "expenses" in t or "seasonal" in t) and ("driven" in t or "because" in t or "due to" in t or "as a result of" in t or "despite" in t or "suggest" in t or "impacted by" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif "refin" in t and ("changes" in t or "flexibility" in t or "maintenance" in t or "configuration" in t or "throughput" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif "largest" in t and ("importer" in t or "exporter" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif "yield" in t and ("increas" in t or "decreas" in t or "because" in t or "due to" in t or "as a result of" in t or "despite" in t or "suggest" in t or "impacted by" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif "flexibility" in t and ("medium" in t or "heavy" in t or "sour" in t or "sweet" in t and "%" in t or "term" in t or "spot" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif "expanded" in t and "marketing" in t:
        operational.append(tokens[i])
        indoperational.append(i)
    elif ("increased" in t or "retail" in t or "branded" in t or "opened" in t) and ("store" in t or "outlet" in t or "station" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif ("price" in t or "demand" in t or "supply" in t or "sales" in t or "cost" in t or "volume" in t or "saving" in t or "inventor" in t) and ("driven" in t or "because" in t or "due to" in t or "as a result of" in t or "despite" in t or "suggest" in t or "impacted by" in t):
        operational.append(tokens[i])
        indoperational.append(i)
    elif ("gasoline" in t or "kerosene" in t or "fuel oil" in t or "sour" in t or "sweet" in t or "medium" in t or "heavy" in t or "light" in t or "basket" in t or "term" in t or "spot" in t) and "%" in t:
        operational.append(tokens[i])
        indoperational.append(i)
    elif "non-cash" in t and "tax" in t:
        operational.append(tokens[i])
        indoperational.append(i)
    elif "source" in t and "from" in t:
        operational.append(tokens[i])
        indoperational.append(i)
    
    
    
        
#print(capex)
print(indoperational)
len(operational)

operational = remove_less_than_six(operational)
operational = remove_unimportant(operational)
for i,m in enumerate(operational):
    operational[i] = replace_char(m)

##Checking irrelevant tokens##
##Irrelevant tokens##
irrelevant=[]
indirrelevant=[]
for i in range(0,(len(tokens))):
    if i in marketind or i in indcapex or i in indoperational:
        continue
    print(tokens[i])
    irrelevant.append(tokens[i])
    indirrelevant.append(i)

len(irrelevant) 
    
##Writing all the 3 sections in one word doc##
#os.chdir('C:\\Users\\shyamal.banerjee\\Desktop\\Meeting 12.07.2018')
os.chdir('\\\\sidc1isln03\\RIL\\PRIVATE\\A&SI_TEAM\\#4.0 Finance & Risk Analytics\\NLP-summarisation problem\\Summary reports')
write_file=filenames[int(yourvar)]
text_file = open(write_file,"w+")
text_file.write("\n        Market Outlook and Business Environment \n\n\n")
for item in marketoutlook:
  text_file.write("%s\n\n" % item)
text_file.write("\n        Operational and Business Highlights \n\n\n")
for item in operational:
    text_file.write("%s\n\n" % item)
text_file.write("\n        Capex and Investment Plans \n\n\n")
for item in capex:
    text_file.write("%s\n\n" % item)
text_file.write("\n        Irrelevant \n\n\n")
for item in irrelevant:
    text_file.write("%s\n\n" % item)
text_file.close()    


# =============================================================================
# for w in market_wordslist:
#     if w in marketoutlook[2]:
#         print(w) 
# for w in capex_wordslist:
#     if w in capex[1]:
#         print(w)
# =============================================================================


        