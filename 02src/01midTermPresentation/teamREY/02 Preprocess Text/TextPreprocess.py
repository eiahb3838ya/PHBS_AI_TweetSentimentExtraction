# -*- coding: utf-8 -*-
"""
Created on Sat May  2  2020

@author: Robert, Edward
"""
try:
    from textblob import TextBlob
    import spacy
    import itertools
    import re
    import unidecode
    from word2number import w2n
except:
    print("in console, use 'pip install -U' for textbolb, spacy, re, unidecode, word2number")
    

class TextPreprocess(object):
    """
    preprocess a single text e.g. "I'd like to have two cups of tea"
    """
    def __init__(self, model = "en_core_web_sm", returnToken = True):
        self.nlp = spacy.load(model)
        self.returnToken = returnToken
        
    def set_text(self, text):
        self.text = text
    
    def check_pos(self, text):
        for token in self.nlp(text):
            print("token: {}, pos: {}, explain: {}.\n".format(token.text,token.pos_,spacy.explain(token.pos_)))
    
    def set_returnTokenStatus(self, status):
        self.returnToken = status
        # print return token or return string
        if self.returnToken:
            print("rtype: tokenized text. \n")
        else:
            print("rtype: preprocessed string. \n")
        
    def set_deselectStopWords(self, wordList = ["no", "not"]):
        self.deselectStopWords = wordList
        for w in self.deselectStopWords:
            self.nlp.vocab[w].is_stop = False
        print("deselect words: " + " ".join([str(e) for e in wordList]) + ".\n")
        
    def set_newStopWords(self, wordList = []):
        self.newStopWords = wordList
        for w in self.newStopWords:
            self.nlp.vocab[w].is_stop = True
        print("add new stop words: "+ " ".join([str(e) for e in wordList])+ ".\n")
    
    def set_notDeletePunctuation(self, punctList = ["!","?","..."]):
        self.notDeletePunctList = punctList
        print("not delete punctuation list: "+ " ".join([str(e) for e in punctList])+ ".\n")
    
    def set_maximumPunctRepeatition(self, maximumRepeatition):
        self.maximumPunctRepeatition = maximumRepeatition
        print("maximum punctuation repeatition times is {} for allowed punctuations.\n".format(str(self.maximumPunctRepeatition)))
        
    def set_contractionMap(self, contractionMap = {}):
        self.contractionMap = contractionMap
        
    def set_slangDict(self, slangDict = {}):
        self.slangDict = slangDict
    
    def set_preCleaningList(self, preCleaningList = ["#","_"]):
        self.preCleaningList = preCleaningList
        print("pre-cleaning list: "+ " ".join([str(e) for e in preCleaningList])+ ".\n")
        
    def auto_setUp(self, deselectWordList = ["no", "not"], 
                   addNewStopWords = [], 
                   contractionMap = {},
                   notDeletePunctuation = ["!","?","..."],
                   maximumPunctRepeatition = 1,
                   slangDict = {},
                   returnTokenStatus = False,
                   preCleaningList = ["#","_"]):
        self.set_deselectStopWords(deselectWordList)
        self.set_newStopWords(addNewStopWords)
        self.set_contractionMap(contractionMap)
        self.set_notDeletePunctuation(notDeletePunctuation)
        self.set_maximumPunctRepeatition(maximumPunctRepeatition)
        self.set_slangDict(slangDict)
        self.set_returnTokenStatus(returnTokenStatus)
        self.set_preCleaningList(preCleaningList)
    
    def add_newContractionMapItem(self, newItems):
        """newItems is a dict"""
        self.contractionMap.update(newItems)
    
    def add_newSlangDictItem(self, newItems):
        """newItems is a dict"""
        self.slangDict.update(newItems)
        
    # process a text/string
    def define_preCleaning(self):
        text = self.text
        if len(self.preCleaningList)>0:
            for char in self.preCleaningList:
                text = re.sub(char," ",text)
        return(text)
    
    def remove_whiteSpace(self):
        """remove extra whitespaces from text"""
        text = self.text.strip()
        return(" ".join(text.split()))
    
    def convert_word2Number(self):
        """convert e.g. three to 3"""
        doc = self.nlp(self.text) #text being tokenized
        tokens = [w2n.word_to_num(token.text) if token.pos_ == 'NUM' else token for token in doc]
        return(" ".join(tokens))
    
    def remove_httpLinks(self):
        """remove http url links"""
        return(re.sub(r"http\S+", "", self.text))

    def remove_accentedNotation(self):
        """remove punct over accented words"""
        text = unidecode.unidecode(self.text)
        return(text)
    
    def correct_spelling(self):
        """can make mistake, e.g. will correct Missin to Kissing"""
        text = TextBlob(self.text).correct()
        return(str(text))
    
    def correct_spelling_oneWord(self, word):
        return("".join("".join(s)[:2] for _,s in itertools.groupby(word)))
    
    def expand_contractionMap(self):
        return(" ".join([self.contractionMap.get(str(e),e) for e in self.text.split(" ")]))
    
    def expand_contractionMap_deprecated(self):
        contractions_pattern = re.compile('({})'.format('|'.join(self.contractionMap.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = self.contractionMap.get(match)\
                                    if self.contractionMap.get(match)\
                                    else self.contractionMap.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return(expanded_contraction)
            
        expanded_text = contractions_pattern.sub(expand_match, self.text)
        expanded_text = re.sub("'", "", expanded_text)
        return(expanded_text)
    
    def expand_slang(self):
        doc = self.nlp(self.text)
        tokens = []
        for token in doc:
            if token.text.upper() in self.slangDict:
                astr = self.slangDict[token.text.upper()]
            else:
                astr = token.text
            tokens.append(astr)
        return(" ".join(tokens))
                
    
    def handle_emotion(self):
        record = self.text
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        record = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', 'EMOTION_SMILE', record)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        record = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', 'EMOTION_LAUGH', record)
        # Love -- <3, :*
        record = re.sub(r'(<3|:\*)', 'EMOTION_LOVE ', record)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        record = re.sub(r'(;-?\)|;-?D|\(-?;)', 'EMOTION_WINK', record)
        # Sad -- :-(, : (, :(, ):, )-:
        record = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', 'EMOTION_SAD', record)
        # Cry -- :,(, :'(, :"(
        record = re.sub(r'(:,\(|:\'\(|:"\()', 'EMOTION_CRY', record)
        return(record)
    
    def lemmatization(self):
        doc = self.nlp(self.text)
        tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
        return(tokens)
    
    # new tokenization method
    def NER_tokenize(self, text):
        doc = self.nlp(text)
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(doc[ent.start:ent.end], attrs={"LEMMA": ent.text})
        return([x for x in doc])
    
    # tokenization
    def tokenize_text(self, remove_accentedChar = True, remove_httpLinks = True, expand_contractionMap = True,
                      handle_emoji = True, convert_word2Number = True, remove_whiteSpace = True, 
                      lemmatization = True,lowercase = True, remove_punctuations = True, 
                      remove_number = True,remove_specialChars = True, expand_slang = True,
                      remove_stopWords = True, correct_spelling_byWord = True, correct_spelling = False):
        # pre cleaning
        self.text = self.define_preCleaning()
        # sentence level preprocess
        if remove_httpLinks:
            self.text = self.remove_httpLinks()
        if remove_whiteSpace:
            self.text = self.remove_whiteSpace()
        if remove_accentedChar:
            self.text = self.remove_accentedNotation()
        if expand_contractionMap:
            self.text = self.expand_contractionMap()
        if lowercase:
            self.text = self.text.lower()
        if correct_spelling:
            self.text = self.correct_spelling()
        if handle_emoji:
            self.text = self.handle_emotion()
        if expand_slang:
            self.text = self.expand_slang()
        
        doc = self.nlp(self.text)
        cleanText = []
        errorLog = []
        
        # init a punctDict
        punctUsageStatsDict = {k:0 for k in self.notDeletePunctList}
        
        # word-level preprocess
        for token in doc:
            flag = True
            editFlag = False
            toEdit = token.text
            edit = ""
            # remove stop words
            try:
                if remove_stopWords and token.is_stop and token.pos_!="NUM":
                    flag = False  
                # remove punctuations
                if remove_punctuations and token.pos_ == "PUNCT" and (toEdit not in self.notDeletePunctList) and flag:
                    flag = False               
                # remove special characters
                if remove_specialChars and token.pos_ in ["SYM","ADP","X","AUX"] and flag:
                    flag = False
                # remove numbers
                if remove_number and (token.pos_=="NUM" or toEdit.isnumeric()) and flag:
                    flag = False
                # convert word to numbers
                if convert_word2Number and token.pos_ == "NUM" and flag:
                    edit = w2n.word_to_num(toEdit)      
                # lemmatization
                if lemmatization and token.is_alpha and token.lemma_ != "-PRON-" and flag:
                    edit = token.lemma_
                    editFlag = True
                # correct spelling by word
                if correct_spelling_byWord and token.is_alpha and token.lemma_ != "-PRON-" and flag:
                    edit = self.correct_spelling_oneWord(token.lemma_)
                    editFlag = True
                # constraint repeatition times of allowed punctuations
                if token.pos_ == "PUNCT" and toEdit in self.notDeletePunctList and flag:
                    punctUsageStatsDict[toEdit] += 1
                    if punctUsageStatsDict[toEdit]>self.maximumPunctRepeatition:
                        flag = False                
                # append valid result to cleanText
                if toEdit!="" and flag and editFlag:
                    cleanText.append(str(edit))
                if toEdit!="" and flag and (not editFlag):
                    cleanText.append(str(toEdit))
            except:
                errorLog.append(" ".join([str(e) for e in doc]))
                break
    
        doc = self.NER_tokenize(" ".join(cleanText)) #re-tokenization
        
        if self.returnToken:
            return([str(e) for e in doc], errorLog)
        else:
            return(" ".join([str(e) for e in doc]), errorLog)
            