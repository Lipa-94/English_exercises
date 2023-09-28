import re
import random

import pandas as pd
import numpy as np
import gensim
import spacy
import pyinflect
import streamlit  as st

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import gensim.downloader as api

np.random.seed(123)
random.seed(123)

class ExerciseGen():
    
    def __init__(self):
        """Initiation of ExerciseGen() object. 
        Contain spacy 'en_core_web_sm' model and gensim 'glove-wiki-gigaword-100' model"""
        
        # Small spacy model
        self.__nlp = spacy.load("en_core_web_sm")

        # Small glove wiki model
        # Attention - it takes a very long time to download if it is not already installed
        self.__model = api.load("glove-wiki-gigaword-100")
        
        # Fix random seed
        np.random.seed(123)
        random.seed(123)
    
    
    def open_text(self, text):
        """Split text by paragraphs and create dataframe from text.
        
        Parameters
        ----------
        text : str - original text for exercise generator
        
        Returns
        -------
        pd.DataFrame() with column 'raw'. 1 row contain one sentence from original text"""
        
        # Convert text to DataFrame
        paragraphs = text.split("\n")  # Splitting text into paragraphs
        dataset = pd.DataFrame({'raw': paragraphs})
        dataset = dataset[dataset['raw'] != '']

        # Split text in DataFrame by sentences
        rows_list = []
        list_text = dataset['raw'].values
        for i in list_text:
            doc = self.__nlp(i)
            a = [sent.text.strip() for sent in doc.sents]
            for j in a:
                rows_list.append(j)
        df = pd.DataFrame(rows_list, columns=['raw'])

        return df
    
    
    def open_file(self, file):
        """Split text by paragraphs and create dataframe from csv/text file.
        
        Parameters
        ----------
        file : file - csv or text file which contains original text for exercise generator
        
        Returns
        -------
        pd.DataFrame() with column 'raw'. 1 row contain one sentence from original text
        """
        
        # Convert text to DataFrame
        dataset = pd.read_csv(file, names=['raw'], delimiter="\t")

        # Split text in DataFrame by sentences
        rows_list = []
        list_text = dataset['raw'].values
        for i in list_text:
            doc = self.__nlp(i)
            a = [sent.text.strip() for sent in doc.sents]
            for j in a:
                rows_list.append(j)
        df = pd.DataFrame(rows_list, columns=['raw'])

        return df
    
    
    def quotes_func(self, text):
        """Add opening or closing quote to sentence, which has odd number of quotes
        
        Parameters
        ----------
        text : str - text that could have odd number of quotes
        
        Returns
        -------
        str with text that has even number of quotes
        """
        
        # count_q - all quotes, count_q_open - opening quotes
        count_q = 0
        count_q_open = 0

        # Count quotes in string. If next char is letter - that is opening quote
        for i in range(len(text)):
            if text[i] == '"':
                count_q += 1
                try:
                    if re.sub(r'[^a-zA-Z]', '', text[i+1]) != '':
                        count_q_open += 1 
                except:
                    pass

        # Count closing quotes
        count_q_close = count_q - count_q_open

        # Return the same text if number of opening and closing quotes equal
        if count_q_close == count_q_open:
            return text
        # Add opening quote if the number of opening quotes less than the number of closing ones
        elif count_q_close > count_q_open:
            return '"'+text
        # Add closing quote if the number of opening quotes more than the number of closing ones
        else:
            return text+'"'
        

    def beautify_text(self, df):
        """Add missing quotes and concatenate broken rows
        
        Parameters
        ----------
        df: pd.DataFrame() - dataframe with text that need to be improved
        
        Returns
        -------
        pd.DataFrame() with improved text
        """
        
        # If row starts with lowercase letter, join it with the previous row
        # because that is ending of direct speech
        df['shift_raw'] = df['raw'].shift(periods=-1, fill_value='')
        df['raw_regex'] = df['raw'].str[:1].str.replace(r'[^a-z]', '', regex=True)
        df['shift_raw_regex'] = df['raw_regex'].shift(periods=-1, fill_value='')
        df.loc[df['shift_raw_regex'] != '', 'raw'] = (df.loc[df['shift_raw_regex'] != '', 'raw'] + 
                                                      '" ' + 
                                                      df.loc[df['shift_raw_regex'] != '', 'shift_raw'])
        df = df[df['raw_regex'] == '']
        df = df.drop(['shift_raw', 'raw_regex', 'shift_raw_regex'], axis=1)

        # Add missing quotes
        df['raw'] = df['raw'].apply(self.quotes_func)

        # Add column with index
        df = df.reset_index(drop=True)
        df = df.reset_index()
        df = df.rename(columns={"index": "row_num"})

        return df
    
    
    def select_word_syn_ant(self, text, pos=['NOUN', 'VERB', 'ADJ', 'ADV'], q_words=1):
        """Create english exercise: select correct missing word from synonym, antonym and correct word.
        
        Parameters
        ---------- 
        - text : str - text with words for exercise
        - pos : list() - array with parts' of speech names. Default variant contain noun, verb, advective and adverb
        - q_words : int - number of words that will be questioned   
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        
        Function has a problem: sometimes it returns too similar words
        """
        
        task_type = 'select_word_syn_ant'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильное слово'

        # Save all available words with type in 'pos' arg. Later will be choosen only 'q_words' number of words
        # For each token save into task_object: token text, the index of the beginning and ending of the token in the text
        save_text = text
        for token in self.__nlp(text):
            if token.pos_ in pos:
                index = save_text.find(token.text)
                task_object.append([token, index, index+len(token.text)])
                save_text = save_text[:index] + '#'*len(token.text) + save_text[index+len(token.text):]
                
        # If we found more than 1 word with questioned type, create an exercise
        if len(task_object) >= 1:
            
            # Choose random elements in quantity 'q_words'
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            task_object.sort(key=lambda x:x[1])
            
            # Replace all random choosen tokens with '_____'
            lag = 0
            for token, index_start, index_end in task_object:
                task_text = task_text[:index_start-lag] + '_____' + task_text[index_end-lag:]
                lag += len(token.text) - 5 # because 5 underscores
                task_answer.append(token.text)
                task_result.append('')
            
            task_object = [token.text for token, index_start, index_end in task_object]
            task_options = [[token] for token in task_object]

            for token in task_options:
                token_new = token[0].lower()

                # Find synonyms
                synonyms = self.__model.similar_by_word(token_new)
                synonyms = [ _[0] for _ in synonyms]
                synonyms = [_.text for _ in self.__nlp(' '.join(synonyms)) if not _.is_stop]
                try:
                    if synonyms[0] == token_new:
                        token.append(synonyms[1].title() if token[0].istitle() else synonyms[1])
                    else:
                        token.append(synonyms[0].title() if token[0].istitle() else synonyms[0])
                except:
                    pass

                # Find antonyms
                antonyms = self.__model.most_similar(positive=[token_new, 'bad'], negative=['good'])
                antonyms = [ _[0] for _ in antonyms]
                antonyms = [_.text for _ in self.__nlp(' '.join(antonyms)) if not _.is_stop]
                try:
                    if antonyms[0] == token_new or antonyms[0].title() == token[1] or antonyms[0] == token[1]:
                        token.append(antonyms[1].title() if token[0].istitle() else antonyms[1])
                    else:
                        token.append(antonyms[0].title() if token[0].istitle() else antonyms[0])
                except:
                    pass

                random.shuffle(token)                
                
        # If there are no words with type in 'pos' array, return empty exercise
        else:
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
    

    def select_word_adj(self, text, q_words=1):
        """Create english exercise: select correct form of adjective.

        Parameters
        ----------  
        - text: text with words for exercise
        - q_words: number of words that will be questioned 

        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        """

        task_text = text
        task_type = 'select_word_adj'
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильную форму прилагательного'
        
        # Save all available adjectives with 3 form available. Later will be choosen only 'q_words' number of words
        # For each token save into task_object: token text, the index of the beginning and ending of the token in the text
        save_text = text
        for token in self.__nlp(text):
            # Find adjective with 3 available forms
            if (token.pos_=='ADJ' and 
                token._.inflect('JJ') != None and 
                token._.inflect('JJR') != None and 
                token._.inflect('JJS') != None):
                index = save_text.find(token.text)
                task_object.append([token, index, index+len(token.text)])
                save_text = save_text[:index] + '#'*len(token.text) + save_text[index+len(token.text):]
        
        # If we found more than 1 word with questioned type, create an exercise
        if len(task_object) > 0:
            
            # Choose random elements in quantity 'q_words'
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            task_object.sort(key=lambda x:x[1])
            
            # Replace all random choosen tokens with '_____'
            lag = 0
            for token, index_start, index_end in task_object:
                task_text = task_text[:index_start-lag] + '_____' + task_text[index_end-lag:]
                lag += len(token.text) - 5 # because 5 underscores
                task_answer.append(token.text)
                task_result.append('')
            
            task_object = [token for token, index_start, index_end in task_object]
            
            for token in task_object:
                task_adv_options = []
                task_adv_options.append(token._.inflect('JJ'))  # JJ      Adjective
                task_adv_options.append(token._.inflect('JJR')) # JJR     Adjective, comparative
                task_adv_options.append(token._.inflect('JJS')) # JJS     Adjective, superlative
                task_options.append(task_adv_options)
            
            task_object = [token.text for token in task_object]

        # If text do not contain adjective with 3 available forms, return empty exercise
        else:
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
    

    def select_word_verb(self, text, q_words=2):
        """Create english exercise: select correct verb form.
        
        Parameters
        ---------- 
        - text: text with words for exercise
        - q_words: number of words that will be questioned
        
        Returns
        ------- 
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        """
        
        task_type = 'select_word_verb'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильную форму глагола'
        
        # Save all available verbs. Later will be choosen only 'q_words' number of words
        # For each token save into task_object: token text, the index of the beginning and ending of the token in the text
        save_text = text
        for token in self.__nlp(text):
            if token.pos_=='VERB':
                index = save_text.find(token.text)
                task_object.append([token, index, index+len(token.text)])
                save_text = save_text[:index] + '#'*len(token.text) + save_text[index+len(token.text):]
                
        # If we found more than 1 word with questioned type, create an exercise
        if len(task_object) > 0:
            
            # Choose random elements in quantity 'q_words'
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            task_object.sort(key=lambda x:x[1])
            
            # Replace all random choosen tokens with '_____'
            lag = 0
            for token, index_start, index_end in task_object:
                task_text = task_text[:index_start-lag] + '_____' + task_text[index_end-lag:]
                lag += len(token.text) - 5 # because 5 underscores
                task_answer.append(token.text)
                task_result.append('')
            
            task_object = [token for token, index_start, index_end in task_object]
            
            for token in task_object:
                task_adv_options = []
                for i in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
                    # VB      Verb, base form
                    # VBD     Verb, past tense
                    # VBG     Verb, gerund or present participle
                    # VBN     Verb, past participle
                    # VBP     Verb, non-3rd person singular present
                    # VBZ     Verb, 3rd person singular present
                    # MD      Modal
                    if token._.inflect(i) not in task_adv_options and token._.inflect(i) != None:
                        task_adv_options.append(token._.inflect(i)) 
                task_options.append(task_adv_options)
            
            task_object = [token.text for token in task_object]

        if task_object == []:
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
    
    
    def select_sent_word(self, text, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1):
        """Create english exercise: select correct sentence from 3 available. 
        Some words in incorrect sentences are replaced with synonym or antonym.
        
        Parameters
        ---------- 
        - text: text with words for exercise
        - pos: array with parts' of speech names
        - q_words: number of words that will be questioned        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        
        Function has a problem: sometimes it returns too similar words
        """
        
        task_type = 'select_sent_word'
        task_text = text
        task_object = [text]
        task_options = [text]
        task_answer = [text]
        task_result = ['']
        task_description = 'Выберите правильное предложение'
        
        # Save all tokens and their beginning and ending index
        save_text = text
        tokens = []
        for token in self.__nlp(text):
            if token.pos_ in pos:
                index = save_text.find(token.text)
                tokens.append([token, index, index+len(token.text)])
                save_text = save_text[:index] + '#'*len(token.text) + save_text[index+len(token.text):]
                  
        # If we found more than 1 token, create an exercise. If text is too long, the full text will exceed max size of window, so we create an exercise only for short texts
        if len(tokens) > 0 and len(text) < 100:
            
            # Choose random elements in quantity 'q_words'
            tokens = random.sample(tokens, k=min(q_words, len(tokens)))
            tokens.sort(key=lambda x:x[1])
            
            # Create 2nd sentence with synonyms and 3rd sentence with antonyms
            second_sentence = text
            scnd_lag = 0
            third_sentence = text
            thrd_lag = 0
            i=5
            for token, start_index, end_index in tokens:
                m, n = np.random.randint(0, i, 2)
                synonym = self.__model.most_similar(token.text.lower(), topn=i)[m][0]
                synonym = synonym.title() if token.text.istitle() else synonym
                second_sentence = second_sentence[:start_index+scnd_lag] + synonym + second_sentence[end_index+scnd_lag:]
                scnd_lag += len(synonym) - len(token.text)
                
                antonym = self.__model.most_similar(positive = [token.text.lower(), 'bad'],
                                                negative = ['good'],
                                                topn=i)[n][0]
                antonym = antonym.title() if token.text.istitle() else antonym
                third_sentence = third_sentence[:start_index+thrd_lag] + antonym + third_sentence[end_index+thrd_lag:]
                thrd_lag += len(antonym) - len(token.text)
            
            task_options.append(second_sentence)
            task_options.append(third_sentence)
            random.shuffle(task_options)

        else:
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
    
    
    def select_sent_adj(self, text, q_words=1):
        """Create english exercise: select correct sentence from 3 available. 2nd and 3rd sentence have adjective in incorrect form.
        
        Parameters
        ---------- 
        - text: text with words for exercise
        - q_words: number of words that will be questioned        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        """
        
        task_type = 'select_sent_adj'
        task_text = text
        task_object = [text]
        task_options = [text]
        task_answer = [text]
        task_result = ['']
        task_description = 'Выберите предложение с правильной формой прилагательного'
        
        # Save all verbs and their beginning and ending index
        save_text = text
        adjs = []
        for token in self.__nlp(text):
            # Find adjective with 3 available forms
            if (token.pos_=='ADJ' and 
                token._.inflect('JJ') != None and 
                token._.inflect('JJR') != None and 
                token._.inflect('JJS') != None):
                index = save_text.find(token.text)
                adjs.append([token, index, index+len(token.text)])
                save_text = save_text[:index] + '#'*len(token.text) + save_text[index+len(token.text):]
        
        # If we found more than 1 verb, create an exercise. If text is too long, the full text will exceed max size of window, so we create an exercise only for short texts
        if len(adjs) > 0 and len(text) < 100:
            
            # Choose random elements in quantity 'q_words'
            adjs = random.sample(adjs, k=min(q_words, len(adjs)))
            adjs.sort(key=lambda x:x[1])
            
            adj_forms = []
            for token, start_index, end_index in adjs:
                token_adj_forms = []
                for j in ['JJ', 'JJR', 'JJS']:
                    # JJ      Adjective
                    # JJR     Adjective, comparative
                    # JJS     Adjective, superlative
                    if token._.inflect(j) != token.text and token._.inflect(j) != None and token._.inflect(j) not in token_adj_forms:
                        token_adj_forms.append(token._.inflect(j))
                adj_forms.append(token_adj_forms)
            
            for _ in range(2):
                new_word_1 = []
                new_sent_1 = text
                lag = 0
                for i, adj in enumerate(adjs):
                    token, start_index, end_index = adj
                    new_word_1.append(random.choice(adj_forms[i]))
                    adj_forms[i].remove(new_word_1[i])
                    new_word_1[i].title() if token.text.istitle() else new_word_1[i]
                    new_sent_1 = new_sent_1[:start_index-lag] + new_word_1[i] + new_sent_1[end_index-lag:]
                    lag += len(token.text) - len(new_word_1[i])
                task_options.append(new_sent_1)
            random.shuffle(task_options)
            
        else:
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
    
    
    def select_sent_verb(self, text, q_words=1):
        """Create english exercise: select correct sentence from 3 available. 2nd and 3rd sentence have verb in incorrect form.
        
        Parameters
        ---------- 
        - text: text with words for exercise
        - q_words: number of words that will be questioned        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        """
        
        task_type = 'select_sent_verb'
        task_text = text
        task_object = [text]
        task_options = [text]
        task_answer = [text]
        task_result = ['']
        task_description = 'Выберите предложение с правильной формой глагола'
        
        # Save all verbs and their beginning and ending index
        save_text = text
        verbs = []
        for token in self.__nlp(text):
            if token.pos_=='VERB':
                index = save_text.find(token.text)
                verbs.append([token, index, index+len(token.text)])
                save_text = save_text[:index] + '#'*len(token.text) + save_text[index+len(token.text):]
        
        # If we found more than 1 verb, create an exercise. If text is too long, the full text will exceed max size of window, so we create an exercise only for short texts
        if len(verbs) > 0 and len(text) < 100:
            
            # Choose random elements in quantity 'q_words'
            verbs = random.sample(verbs, k=min(q_words, len(verbs)))
            verbs.sort(key=lambda x:x[1])
            
            verb_forms = []
            for token, start_index, end_index in verbs:
                token_verb_forms = []
                for j in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
                    # VB      Verb, base form
                    # VBD     Verb, past tense
                    # VBG     Verb, gerund or present participle
                    # VBN     Verb, past participle 
                    # VBP     Verb, non-3rd person singular present
                    # VBZ     Verb, 3rd person singular present
                    # MD      Modal
                    if token._.inflect(j) != token.text and token._.inflect(j) != None and token._.inflect(j) not in token_verb_forms:
                        token_verb_forms.append(token._.inflect(j))
                verb_forms.append(token_verb_forms)
            
            for _ in range(2):
                new_word_1 = []
                new_sent_1 = text
                lag = 0
                for i, verb in enumerate(verbs):
                    token, start_index, end_index = verb
                    new_word_1.append(random.choice(verb_forms[i]))
                    verb_forms[i].remove(new_word_1[i])
                    new_word_1[i].title() if token.text.istitle() else new_word_1[i]
                    new_sent_1 = new_sent_1[:start_index-lag] + new_word_1[i] + new_sent_1[end_index-lag:]
                    lag += len(token.text) - len(new_word_1[i])
                task_options.append(new_sent_1)
            random.shuffle(task_options)
            
        else:
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
        
    
    def select_memb_groups(self, text, q_words=1):
        """Create english exercise: select correct name of part of speech for the part of text highlighted in bold.
        
        Parameters
        ---------- 
        - text: text with words for exercise
        - q_words: number of text parts that will be questioned        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        """
        
        task_type = 'select_memb_groups'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Определите, чем является выделенное словосочетание'
        
        # Save all chunks
        save_text = text
        for chunk in self.__nlp(text).noun_chunks:
            index = save_text.find(chunk.text)
            task_object.append([chunk, index, index+len(chunk.text)])
            save_text = save_text[:index] + '#'*len(chunk.text) + save_text[index+len(chunk.text):]

        # Text must have more than one chuck
        if len(task_object) > 1:
            
            # Choose random elements in quantity 'q_words'
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            task_object.sort(key=lambda x:x[1])
            
            # Highlight chunk in bold
            lag = 0
            for chunk, index_start, index_end in task_object:
                task_text = task_text[:index_start+lag] + '**' + task_text[index_start+lag:index_end+lag]+ '**' + task_text[index_end+lag:]
                lag += 4 # because 4 asterisks
                task_answer.append(spacy.explain(chunk.root.dep_))
                task_options.append('')
                task_result.append('')
            
            task_object = [chunk.text for chunk, index_start, index_end in task_object]

            # All chunks from text forms all available task answers
            # If we have less than 3 available answers then fill them with random vaues from dep_list
            unique_answers = list(set(task_answer))
            dep_list = ['clausal modifier of noun (adjectival clause)', 'adjectival complement', 'adverbial clause modifier', 
                    'adverbial modifier', 'agent', 'adjectival modifier', 'appositional modifier', 'attribute', 'auxiliary', 
                    'auxiliary (passive)', 'case marking', 'coordinating conjunction', 'clausal complement', 'compound', 
                    'conjunct', 'copula', 'clausal subject', 'clausal subject (passive)', 'dative', 'unclassified dependent', 
                    'determiner', 'direct object', 'expletive', 'interjection', 'marker', 'meta modifier', 'negation modifier', 
                    'noun compound modifier', 'noun phrase as adverbial modifier', 'nominal subject', 'nominal subject (passive)', 
                    'object predicate', 'object', 'oblique nominal', 'complement of preposition', 'object of preposition', 
                    'possession modifier', 'pre-correlative conjunction', 'prepositional modifier', 'particle', 'punctuation', 
                    'modifier of quantifier', 'relative clause modifier', 'root', 'open clausal complement']
            for i in unique_answers:
                try:
                    dep_list.remove(i)
                except:
                    pass
            if len(unique_answers) == 2:
                unique_answers.append(random.choice(dep_list))
            elif len(unique_answers) == 1:
                unique_answers.extend(random.sample(dep_list, k=2))
            random.shuffle(unique_answers)
            task_options = [unique_answers for _ in task_options]

            # If text has only one chunk, return empty exercise
        else:
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
    
    
    def fill_words_in_the_gaps(self, text, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1, hint=True):
        """Create english exercise: fill missing word. If hint=True, first letter of missing word in known
        
        Parameters
        ---------- 
        - text: text with words for exercise
        - pos: array with parts' of speech names
        - q_words: number of words that will be questioned
        - hint: if True, first letter of missing word will be shown. If False - first letter will be hidden
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        """
        
        task_type = 'fill_words_in_the_gaps'
        task_text = text     
        task_object = []     
        task_options = []    
        task_answer = []     
        task_result = []     
        task_description = 'Заполните пропущенное слово (вместе с первой буквой, если была использована подсказка)'
        
        # Save all tokens and their beginning and ending index
        save_text = text
        tokens = []
        for token in self.__nlp(text):
            if token.pos_ in pos:
                index = save_text.find(token.text)
                tokens.append([token, index, index+len(token.text)])
                save_text = save_text[:index] + '#'*len(token.text) + save_text[index+len(token.text):]
        
        # If we found more than q_words+3 tokens, create an exercise. If there will be less than q_words+3 tokens, it would be hard to guess words
        if len(tokens) > q_words+3:
            
            # Choose random elements in quantity 'q_words'
            tokens = random.sample(tokens, k=min(q_words, len(tokens)))
            tokens.sort(key=lambda x:x[1])
            
            lag = 0
            task_text = text
            for token, start_index, end_index in tokens:
                task_object.append(token.text)
                task_answer.append(token.text)
                task_result.append('')
                
                if hint:
                    task_text = task_text[:start_index-lag] + token.text[0] + '_____' + task_text[end_index-lag:]
                    lag += len(token.text) - 6
                else:
                    task_text = task_text[:start_index-lag] + '_____' + task_text[end_index-lag:]
                    lag += len(token.text) - 5

        else:
            return {'raw' : text,
                    'task_type' : task_type,
                    'task_text' : text,
                    'task_object' : np.nan,
                    'task_options' : np.nan,
                    'task_answer' : np.nan,
                    'task_result' : np.nan,
                    'task_description' : np.nan,
                    'task_total': np.nan
                   }
        
        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
               }
    
    
    def listening_fill_chunks(self, text, q_words=1):
        """Create english exercise: fill missing chunk
        
        Parameters
        ---------- 
        - text: text with chunks for exercise
        - q_words: number of chunks that will be questioned        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        
        Function has a problem: sometimes it returns too similar words
        """
        
        task_type = 'listening_fill_chunks'
        task_text = text     
        task_object = []     
        task_options = []    
        task_answer = []     
        task_result = []     
        task_description = 'Прослушайте аудиозапись и заполните пропущенное словосочетание'
        
        # Save all chunks
        save_text = text
        for chunk in self.__nlp(text).noun_chunks:
            index = save_text.find(chunk.text)
            task_object.append([chunk, index, index+len(chunk.text)])
            save_text = save_text[:index] + '#'*len(chunk.text) + save_text[index+len(chunk.text):]
        
        # If we found more than 1 chunk, create an exercise. If there will be less than 1 chunk, it would be hard to guess words
        if len(task_object) > 1:
            
            # Choose random elements in quantity 'q_words'
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            task_object.sort(key=lambda x:x[1])
            
            lag = 0
            task_text = text
            for chunk, start_index, end_index in task_object:
                task_answer.append(chunk.text)
                task_result.append('')
                task_text = task_text[:start_index-lag] + '_____' + task_text[end_index-lag:]
                lag += len(chunk.text) - 5
            
            task_object = [chunk.text for chunk, start_index, end_index in task_object]

        # If text has only one chunk, return empty exercise
        else:
            return {'raw' : text,
                    'task_type' : task_type,
                    'task_text' : text,
                    'task_object' : np.nan,
                    'task_options' : np.nan,
                    'task_answer' : np.nan,
                    'task_result' : np.nan,
                    'task_description' : np.nan,
                    'task_total': np.nan
                   }
        
        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
               }
    
    
    def set_word_order(self, text):
        """Create english exercise: all the words in sentence are mixed up
        
        Parameters
        ---------- 
        - text: text with words for exercise  
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : List(), 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        """
        
        task_type = 'set_word_order'
        task_text = text.split(' ')
        random.shuffle(task_text)
        task_object = []
        task_options = []
        task_answer = [text.split(' ')]
        task_result = ['']
        task_description = 'Расставьте слова в правильном порядке'
       
        # Do not generate exersice if there to small or too large number of words
        if  len(task_text) < 3 or len(task_text) > 10:
            task_text = np.nan
            task_object = np.nan
            task_options = np.nan
            task_answer = np.nan
            task_result = np.nan
            task_description = np.nan

        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description,
                'task_total': 0
                }
    
    
    def sent_with_no_exercises(self, text):
        """Create empty english exercise.
        We use this type of exercise only then no one else type of exercise is available
        
        Parameters
        ----------  
        - text: text to show on display        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : np.nan, 'task_options' : np.nan, 
         'task_answer' : np.nan, 'task_result' : np.nan, 'task_description' : np.nan, 'task_total': int}
        """
        
        return {'raw' : text,
                'task_type' : 'sent_with_no_exercises',
                'task_text' : text,
                'task_object' : np.nan,
                'task_options' : np.nan,
                'task_answer' : np.nan,
                'task_result' : np.nan,
                'task_description' : 'Предложение без упражнения',
                'task_total': np.nan
                }
    

    def create_lesson(self, 
                      df, 
                      start_row=1, 
                      q_task=20, 
                      list_of_exercises=[True, True, True, True, True, True, True, True, True, True], 
                      q_words=[1, 1, 1, 1, 1, 1, 1, 1, 1]):
        """Create english lesson from dataframe. Default lesson starts from 1st sentence and include 20 exercises 
        of all types with only one missing word/chunk into each of them. For each sentence in range creates all possible exercises.
        
        Parameters
        ----------  
        - df: dataframe, which contains only text and row_number
        - start_row: the number of first sentence to start exercise generator
        - q_task: task quantity
        - list_of_exercises: contains list with bools. Every element of list means certain type of exercise: 
        select_word_syn_ant, select_word_adj, select_word_verb, select_memb_groups, 
        select_sent_verb, select_sent_word, fill_words_in_the_gaps. If element is False, that type of exercise will be banned
        - q_words: number of words/chunks to replace in original text 
        
        Returns
        -------
        pd.DataFrame with english exercises
        """
        
        q_task_fact = 0
        lesson_tasks = pd.DataFrame(columns=['row_num', 'raw', 'task_type', 'task_text', 'task_object', 'task_options', 
                                      'task_answer', 'task_result', 'task_description', 'task_total'])

        # For each row in dataframe save all available exercises
        for i in range(start_row-1, len(df)):
            mark = 0
            if q_task_fact < q_task:
                row_tasks = pd.DataFrame(columns=['raw', 'task_type', 'task_text', 'task_object', 'task_options', 
                                                  'task_answer', 'task_result', 'task_description', 'task_total'])
                if list_of_exercises[0]:
                    row_tasks.loc[mark] = self.select_word_syn_ant(df.loc[i, 'raw'], q_words=q_words[0])
                    mark += 1
                if list_of_exercises[1]:
                    row_tasks.loc[mark] = self.select_word_adj(df.loc[i, 'raw'], q_words=q_words[1])
                    mark += 1
                if list_of_exercises[2]:
                    row_tasks.loc[mark] = self.select_word_verb(df.loc[i, 'raw'], q_words=q_words[2])
                    mark += 1
                if list_of_exercises[3]:
                    row_tasks.loc[mark] = self.select_sent_word(df.loc[i, 'raw'], q_words=q_words[3])
                    mark += 1
                if list_of_exercises[4]:
                    row_tasks.loc[mark] = self.select_sent_adj(df.loc[i, 'raw'], q_words=q_words[4])
                    mark += 1    
                if list_of_exercises[5]:
                    row_tasks.loc[mark] = self.select_sent_verb(df.loc[i, 'raw'], q_words=q_words[5])
                    mark += 1
                if list_of_exercises[6]:
                    row_tasks.loc[mark] = self.select_memb_groups(df.loc[i, 'raw'], q_words=q_words[6])
                    mark += 1
                if list_of_exercises[7]:
                    row_tasks.loc[mark] = self.fill_words_in_the_gaps(df.loc[i, 'raw'], q_words=q_words[7])
                    mark += 1
                if list_of_exercises[8]:
                    row_tasks.loc[mark] = self.listening_fill_chunks(df.loc[i, 'raw'], q_words=q_words[8])
                    mark += 1  
                if list_of_exercises[9]:
                    row_tasks.loc[mark] = self.set_word_order(df.loc[i, 'raw'])
                    mark += 1
                row_tasks.loc[mark] = self.sent_with_no_exercises(df.loc[i, 'raw'])
                # Delete all empty exercises and add row number from original dataframe to save the original order
                row_tasks = row_tasks[row_tasks['raw'].isna() == False]
                row_tasks['row_num'] = df.loc[i, 'row_num']
                row_tasks = row_tasks[row_tasks['task_description'].isna() == False]

                # If any exercise is available, add 1 to counter q_task_fact
                if len(row_tasks[row_tasks['task_type'] != 'sent_with_no_exercises']) != 0:
                    q_task_fact += 1
                lesson_tasks = pd.concat([lesson_tasks, row_tasks], ignore_index=True)

        return lesson_tasks
    
    
    def create_default_lesson(self, df):
        """Shuffle all rows in dataframe and save first row for each row_index for every unique sentense. So, each unique sentence will
        have one random exercise. 
        
        Parameters
        ---------- 
        - df: dataframe, which contains sentences with all possible exercise types 
        
        Returns
        -------
        pd.DataFrame with english exercises
        """
        
        new_df = df[df['task_type'] != 'sent_with_no_exercises'].sample(frac=1).sort_values(by='row_num')
        new_df = new_df.groupby('row_num').agg('first').reset_index()

        df_with_no_exer = df[~df['row_num'].isin(new_df['row_num'].unique())]
        new_df = pd.concat([new_df, df_with_no_exer], ignore_index=True).sort_values('row_num').reset_index(drop=True)

        return new_df
    
    
    def show_result_table(self, df):
        """Convert all columns in dataframe into str format and rename columns to show result table in streamlit. 
        
        Parameters
        ---------- 
        - df: dataframe, which contains english exercises with answers 
        
        Returns
        -------
        pd.DataFrame with english exercises and answers
        """
        
        new_df = df[['task_description', 'task_text', 'task_answer', 'task_result', 'task_total']]
        new_df['task_text'] = new_df['task_text'].astype('str')
        new_df['task_answer'] = new_df['task_answer'].astype('str')
        new_df['task_result'] = new_df['task_result'].astype('str')
        new_df.columns = ['Тип задания', 'Текст упражнения', 'Правильный ответ', 'Ответ на задание', 'Результат']
        return new_df
    
    
    def show_result_by_task_type(self, df):
        """Return short result table with all available exercise type and score of correct answers
        
        Parameters
        ---------- 
        - df: dataframe, which contains english exercises with answers
        
        Returns
        -------
        pd.DataFrame with english test results
        """
        
        new_df = (df[df['task_type'] != 'sent_with_no_exercises']
                  .groupby('task_description')['task_total']
                  .agg(['count', 'sum']))
        new_df['wrong_answers'] = new_df['count'] - new_df['sum']
        new_df['result'] = new_df['sum'].astype('int').astype('str') + ' / ' + new_df['count'].astype('int').astype('str')
        new_df = new_df.sort_values(by='wrong_answers', ascending=False)
        new_df = new_df.drop(['count', 'sum', 'wrong_answers'], axis=1).reset_index()
        new_df.columns = ['Тип упражнения', 'Результат']
        return new_df
    
    
    def result_interpretation(self, df):
        """Return string with result interpretation
        
        Parameters
        ---------- 
        - df: dataframe, which contains english exercises with answers
        
        Returns
        -------
        str with result interpretation
        """
        
        correct_cnt = int(df['task_total'].sum())
        all_cnt = int(df.loc[df['task_type'] != 'sent_with_no_exercises', 'task_total'].count())
        correct_prop = round(correct_cnt / all_cnt * 100, 0)
        
        result_info = ('Ваш результат: ' + str(correct_cnt) + ' / ' + str(all_cnt) + ' (' + str(correct_prop) + '%)')
        result_comment = ''
        if correct_prop == 100:
            result_comment = 'А как записаться к вам на урок английского? 🤓'
        elif correct_prop >= 90:
            result_comment = 'Отлично! Нужно совсем немного, чтобы добиться идеального результата 🧗‍♂️'
        elif correct_prop >= 70:
            result_comment = 'Очень неплохой результат! Но в отдельных темах стоит еще потренироваться👨‍🎓'
        else:
            result_comment = 'Стоит повторить несколько тем 🤔'
        
        result_mistakes = ''
        if correct_cnt != all_cnt:
            task_with_mistakes = str(df.loc[df['task_total'] == 0, 'task_description'].unique())
            result_mistakes = ('Ошибки были в упражнениях следующих типов: ' + 
                               task_with_mistakes + 
                               '. В следующий раз попробуй сделать упор именно на такие упражнения')
        
        return result_info, result_comment, result_mistakes
    
    
    def display_dataset(self, df):
        """Function that convert all values in dataframe into type str. This function help to display dataframe while using streamlit
        
        Parameters
        ---------- 
        - df: dataframe, which rows type will be converted to str
        
        Returns
        -------
        pd.DataFrame with str values
        """
        
        new_df = df
        for column in df.columns:
            new_df[column] = new_df[column].astype('str')
        return new_df
    