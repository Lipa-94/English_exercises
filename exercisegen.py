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
        - text : str - text with letters for exercise
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
        for token in self.__nlp(text):
            if token.pos_ in pos:
                task_object.append(token.text)
                
        # If we found more than 1 word with questioned type, create an exercise
        if len(task_object) >= 1:
            # Choose random elements in quantity 'q_words'. Place them in the same order as they came into text
            order = {number:task for number,task in enumerate(task_object)}
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            order_list = []
            for i in task_object:
                order_list.append(list(order.keys())[list(order.values()).index(i)])
            order_list, task_object = zip(*sorted(zip(order_list, task_object)))
            task_object = list(task_object)

            for token in task_object:
                task_text = task_text.replace(token, '_____')
                task_answer.append(token)
                task_result.append('')

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
    
    # Функция создающая упражнение: пропуск прилагательного, и варианты ответа: 3 формы прилагательного
    def select_word_adj(self, text):
        """Create english exercise: select correct form of adjective.
        
        Parameters
        ----------  
        - text: text with letters for exercise
        - q_words: number of words that will be questioned 
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        
        Function has a problem: current version do not have q_words
        """
        
        task_text = text
        task_type = 'select_word_adj'
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильную форму прилагательного'

        for token in self.__nlp(text):
            # Find adjective with 3 available forms
            if (token.pos_=='ADJ' and 
                token._.inflect('JJ') != None and 
                token._.inflect('JJR') != None and 
                token._.inflect('JJS') != None):
                task_text = task_text.replace(token.text, '_____')
                task_object.append(token.text)
                task_answer.append(token.text)
                task_result.append('')
                task_adv_options = []
                task_adv_options.append(token._.inflect('JJ'))  # JJ      Adjective
                task_adv_options.append(token._.inflect('JJR')) # JJR     Adjective, comparative
                task_adv_options.append(token._.inflect('JJS')) # JJS     Adjective, superlative
                task_options.append(task_adv_options)
                
        # If text do not contain adjective with 3 available forms, return empty exercise
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
    

    def select_word_verb(self, text):
        """Create english exercise: select correct verb form.
        
        Parameters
        ---------- 
        - text: text with letters for exercise
        - q_words: number of words that will be questioned
        
        Returns
        ------- 
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
         
        Function has a problem: current version do not have q_words
        """
        
        task_type = 'select_word_verb'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильную форму глагола'

        for token in self.__nlp(text):
            if (token.pos_=='VERB'):
                task_text = task_text.replace(token.text, '_____')
                task_object.append(token.text)
                task_answer.append(token.text)
                task_result.append('')
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
    
    
    def select_memb_groups(self, text, q_words=1):
        """Create english exercise: select correct name of part of speech for the part of text highlighted in bold.
        
        Parameters
        ---------- 
        - text: text with letters for exercise
        - q_words: number of text parts that will be questioned        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        
        Function has a problem: sometimes it returns too similar words
        """
        
        task_type = 'select_memb_groups'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Определите, чем является выделенная фраза'

        # Save all text chunks
        for chunk in self.__nlp(text).noun_chunks:
            task_object.append(chunk)

        # Text must have more than one chuck
        if len(task_object) > 1:
            # Save order of chunks and choose random chunck 
            order = {number:task for number,task in enumerate(task_object)}
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            order_list = []
            for i in task_object:
                order_list.append(list(order.keys())[list(order.values()).index(i)])
            order_list, task_object = zip(*sorted(zip(order_list, task_object)))
            task_object = list(task_object)
            
            # Highlight chunk in bold
            for chunk in task_object:
                task_text = task_text.replace(chunk.text, '**'+chunk.text+'**')
                task_answer.append(spacy.explain(chunk.root.dep_))
                task_object = chunk.text
                task_options.append('')
                task_result.append('')

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
    
    
    def select_sent_verb(self, text):
        """Create english exercise: select correct sentence from 3 available. 2nd and 3rd sentence has verb in incorrect form.
        
        Parameters
        ---------- 
        - text: text with letters for exercise
        - q_words: number of words that will be questioned        
        
        Returns
        -------
        dictionary:
        {'raw' : str, 'task_type' : str, 'task_text' : str, 'task_object' : List(), 'task_options' : List(), 
         'task_answer' : List(), 'task_result' : List(), 'task_description' : str, 'task_total': int}
        
        Function has a problem: current version do not have q_words
        """
        
        task_type = 'select_sent_verb'
        task_text = text
        task_object = [text]
        task_options = [text]
        task_answer = [text]
        task_result = ['']
        task_description = 'Выберите предложение с правильной формой глагола'

        new_sent_1, new_sent_2 = text, text
        i=5
        count_verbs = 0
        for token in self.__nlp(text):
            if (token.pos_=='VERB'):
                # Form list of verb forms which doesn't include original form of verb 
                verb_forms = []
                for i in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
                    # VB      Verb, base form
                    # VBD     Verb, past tense
                    # VBG     Verb, gerund or present participle
                    # VBN     Verb, past participle 
                    # VBP     Verb, non-3rd person singular present
                    # VBZ     Verb, 3rd person singular present
                    # MD      Modal
                    if token._.inflect(i) != token.text and token._.inflect(i) != None and token._.inflect(i) not in verb_forms:
                        verb_forms.append(token._.inflect(i))

                new_word_1 = random.choice(verb_forms)
                new_word_1 = new_word_1.title() if token.text.istitle() else new_word_1
                new_sent_1 = new_sent_1.replace(token.text, new_word_1)
                verb_forms.remove(new_word_1)
                new_word_2 = random.choice(verb_forms)
                new_word_2 = new_word_1.title() if token.text.istitle() else new_word_2
                new_sent_2 = new_sent_2.replace(token.text, new_word_2)
                count_verbs += 1

        if count_verbs > 0 and len(text) < 100:
            task_options.append(new_sent_1)
            task_options.append(new_sent_2)
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
    
    
    def select_sent_word(self, text, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1):
        """Create english exercise: select correct sentence from 3 available. 
        Some words in incorrect sentences are replaced with synonym or antonym.
        
        Parameters
        ---------- 
        - text: text with letters for exercise
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

        q_words_fact = 0
        q_all_words = 0
        new_sent_1, new_sent_2 = text, text
        i=5
        for token in self.__nlp(text):
            q_all_words += 1
            if (token.pos_ in pos) and (q_words_fact < q_words):
                m, n = np.random.randint(0, i, 2)

                new_word_1 = self.__model.most_similar(token.text.lower(), topn=i)[m][0]
                new_word_2 = self.__model.most_similar(positive = [token.text.lower(), 'bad'],
                                                negative = ['good'],
                                                topn=i)[n][0]

                new_word_1 = new_word_1.title() if token.text.istitle() else new_word_1
                new_word_2 = new_word_2.title() if token.text.istitle() else new_word_2

                new_sent_1 = new_sent_1.replace(token.text, new_word_1)
                new_sent_2 = new_sent_2.replace(token.text, new_word_2)
                q_words_fact += 1

        # If sentence has less than 3 tokens or it doesn't cointain word with 'pos' type, return empty exercise
        if q_words_fact > 0 and q_all_words > 3 and len(text) < 100:
            task_options.append(new_sent_1)
            task_options.append(new_sent_2)
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
    
    
    # Функция создающая упражнение: нужно ввести с клавиатуры пропущенное слово
    # Есть выбор, какие части речи пропускать, есть возможность указать долю пропущенных слов, можно использовать подсказку
    def fill_words_in_the_gaps(self, text, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1, hint=True):
        """Create english exercise: fill missing word. If hint=True, first letter of missing word in known
        
        Parameters
        ---------- 
        - text: text with letters for exercise
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
        
        count_words = 0
        count_missing_words = 0
        for token in self.__nlp(text):
            count_words += 1
            if token.pos_ in pos and count_missing_words < q_words:
                if hint:
                    task_text = task_text.replace(token.text, token.text[0]+'____')
                else:
                    task_text = task_text.replace(token.text, '_____')
                count_missing_words += 1
                task_object.append(token.text)
                task_answer.append(token.text)
                task_result.append('')

        if count_missing_words < 1:
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
        else:
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
                      list_of_exercises=[True, True, True, True, True, True, True], 
                      q_words=[1, 1, 1, 1, 1, 1, 1]):
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
                    row_tasks.loc[mark] = self.select_word_adj(df.loc[i, 'raw'])
                    mark += 1
                if list_of_exercises[2]:
                    row_tasks.loc[mark] = self.select_word_verb(df.loc[i, 'raw'])
                    mark += 1
                if list_of_exercises[3]:
                    row_tasks.loc[mark] = self.select_memb_groups(df.loc[i, 'raw'], q_words=q_words[3])
                    mark += 1
                if list_of_exercises[4]:
                    row_tasks.loc[mark] = self.select_sent_verb(df.loc[i, 'raw'])
                    mark += 1
                if list_of_exercises[5]:
                    row_tasks.loc[mark] = self.select_sent_word(df.loc[i, 'raw'], q_words=q_words[5])
                    mark += 1
                if list_of_exercises[6]:
                    row_tasks.loc[mark] = self.fill_words_in_the_gaps(df.loc[i, 'raw'], q_words=q_words[6])
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
        
        return result_info + "\n" + result_comment + "\n" + result_mistakes
    
    
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
    