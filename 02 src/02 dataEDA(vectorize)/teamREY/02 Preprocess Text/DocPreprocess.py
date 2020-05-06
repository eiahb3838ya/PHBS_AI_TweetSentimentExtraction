# -*- coding: utf-8 -*-
"""
Created on Sat May  2  2020

@author: Robert
"""
try:
    from tqdm import tqdm
    from TextPreprocess import TextPreprocess
    # import multiprocessing as mp
except:
    print("in console, use 'pip install -U' for tqdm")
    print("\n")
    print("add TextPreprocess.py to working directory")
    

class DocPreprocess(TextPreprocess):
    """
    Wrap text preprocess method to be applied to a big list
    """
    def __init__(self, doc, model="en_core_web_sm", returnToken = True):
        self.doc = doc
        super().__init__(model = model)
        super().__init__(returnToken = returnToken)
        
    def run_docPreprocessing(self, remove_accentedChar = True, remove_httpLinks = True, expand_contractionMap = True,
                                handle_emoji = True, convert_word2Number = True, remove_whiteSpace = True, 
                                lemmatization = True,lowercase = True, remove_punctuations = True, 
                                remove_number = True,remove_specialChars = True, expand_slang = True,
                                remove_stopWords = True, correct_spelling_byWord = True, correct_spelling = False):
        cleanDoc = []
        errorLogDoc = []
        
        # start loop
        for text in tqdm(self.doc):
            self.set_text(text)
            cleanText, errorLog = self.tokenize_text(remove_accentedChar , remove_httpLinks , expand_contractionMap ,
                                                      handle_emoji , convert_word2Number , remove_whiteSpace , 
                                                      lemmatization ,lowercase , remove_punctuations , 
                                                      remove_number ,remove_specialChars , expand_slang ,
                                                      remove_stopWords , correct_spelling_byWord , correct_spelling) 
            cleanDoc.append(cleanText)
            errorLogDoc.append(errorLog)
            
        return(cleanDoc, errorLogDoc)
    
    #TODO: add parallel module later
    # def run_docPreprocessingParallel(self, remove_accentedChar = True, expand_contractionMap = True,
    #                                   handle_emoji = True, convert_word2Number = True, remove_whiteSpace = True, 
    #                                   lemmatization = True,lowercase = True, remove_punctuations = True, 
    #                                   remove_number = True,remove_specialChars = True, expand_slang = True,
    #                                   remove_stopWords = True, correct_spelling_byWord = True, correct_spelling = False):
    #     
    #     pool = mp.Pool(mp.cpu_count())
    #     cleanDoc, errorLogDoc = [pool.apply(self.tokenize_text, args = (remove_accentedChar , expand_contractionMap ,
    #                                           handle_emoji , convert_word2Number , remove_whiteSpace , 
    #                                           lemmatization ,lowercase , remove_punctuations , 
    #                                           remove_number ,remove_specialChars , expand_slang ,
    #                                           remove_stopWords , correct_spelling_byWord , correct_spelling)) for row in self.doc]
    #     pool.close()    
        
    #     return(cleanDoc, errorLogDoc)
