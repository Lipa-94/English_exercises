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
        # малая модель spacy
        self.__nlp = spacy.load("en_core_web_sm")

        # малая модель glove wiki
        # внимание - очень долго скачивает, если она еще не установлена
        self.__model = api.load("glove-wiki-gigaword-100")
                
        np.random.seed(123)
        random.seed(123)
    
    # Функция загрузки текста с разбивкой на предложения
    def open_text(self, text):
        # Загрузка текста
        paragraphs = text.split("\n")  # Splitting text into paragraphs
        dataset = pd.DataFrame({'raw': paragraphs})
        dataset = dataset[dataset['raw'] != '']

        # Перенос предложений на новые строки
        rows_list = []
        list_text = dataset['raw'].values
        for i in list_text:
            doc = self.__nlp(i)
            a = [sent.text.strip() for sent in doc.sents]
            for j in a:
                rows_list.append(j)
        df = pd.DataFrame(rows_list, columns=['raw'])

        return df
    
    # Функция загрузки файла с разбивкой текста на предложения
    def open_file(self, file):
        # Загрузка текста
        dataset = pd.read_csv(file, names=['raw'], delimiter="\t")

        # Перенос предложений на новые строки
        rows_list = []
        list_text = dataset['raw'].values
        for i in list_text:
            doc = self.__nlp(i)
            a = [sent.text.strip() for sent in doc.sents]
            for j in a:
                rows_list.append(j)
        df = pd.DataFrame(rows_list, columns=['raw'])

        return df
    
    # Функция, которая добавляет открывающую или закрывающую кавычку в текст
    def quots_func(self, text):
        # Считаем кавычки в строке. Сколько всего, сколько открывающих и сколько закрывающих
        count_q = 0
        count_q_open = 0

        # Берем из всей строки последний и предпоследний элемент и считаем количество
        for i in range(len(text)):
            if text[i] == '"':
                count_q += 1
                try:
                    if re.sub(r'[^a-zA-Z]', '', text[i+1]) != '':
                        count_q_open += 1 
                except:
                    pass

        # Считаем, сколько закрывающих кавычек
        count_q_close = count_q - count_q_open

        # Возвращаем строку: если открывающих и закрывающих строк одинаковое количество, возвращаем обычный текст
        if count_q_close == count_q_open:
            return text
        # Если закрывающих скобок больше, чем открывающих, добавляем кавычку в начало
        elif count_q_close > count_q_open:
            return '"'+text
        # Если открывающих скобок больше, чем закрывающих, добавляем кавычку в конец
        else:
            return text+'"'
        
    # Функция, преобразовывающая датасет: корректируется перенос строк, кавычки
    def beautify_text(self, df):
        # Если строка начинаемся с маленькой буквы, присоединяем ее к предыдущей строке, 
        # т.к. скорее всего это завершение прямой речи
        df['shift_raw'] = df['raw'].shift(periods=-1, fill_value='')
        df['raw_regex'] = df['raw'].str[:1].str.replace(r'[^a-z]', '', regex=True)
        df['shift_raw_regex'] = df['raw_regex'].shift(periods=-1, fill_value='')
        df.loc[df['shift_raw_regex'] != '', 'raw'] = (df.loc[df['shift_raw_regex'] != '', 'raw'] + 
                                                      '" ' + 
                                                      df.loc[df['shift_raw_regex'] != '', 'shift_raw'])
        df = df[df['raw_regex'] == '']
        df = df.drop(['shift_raw', 'raw_regex', 'shift_raw_regex'], axis=1)

        # Обработка кавычек
        df['raw'] = df['raw'].apply(self.quots_func)

        # Добавление столбца с индексом
        df = df.reset_index(drop=True)
        df = df.reset_index()
        df = df.rename(columns={"index": "row_num"})

        return df
    
    # Функция создающая упражнение: пропуск слова, и варианты ответа: правильный, синоним, антоним
    # Проблема: иногда выводит слишком похожие слова
    def select_word_syn_ant(self, text, pos=['NOUN', 'VERB', 'ADJ', 'ADV'], q_words=1):
        task_type = 'select_word_syn_ant'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильное слово'

        # Для задания выбираются рандомные слова в количестве q_words
        for token in self.__nlp(text):
            if token.pos_ in pos:
                task_object.append(token.text)

        if len(task_object) > 1:
            # Выбираем случайные элементы и проставляем их в том порядке, в каком они встречались к тексте
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

                # Ищем синонимы
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

                # Ищем антонимы
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
    def select_word_adv(self, text):
        task_text = text
        task_type = 'select_word_adv'
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильную форму прилагательного'

        # изменение степени прилагательного с помощью pyinflect
        for token in self.__nlp(text):
            # выбираем прилагательные, но только те, для которых есть все 3 формы
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
    
    # Функция создающая упражнение: пропуск глагола, и варианты ответа: 3 формы глагола
    def select_word_verb(self, text):
        task_type = 'select_word_verb'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Выберите правильную форму глагола'

        # изменение формы глагола с помощью pyinflect
        for token in self.__nlp(text):
            if (token.pos_=='VERB'):
                task_text = task_text.replace(token.text, '_____')
                task_object.append(token.text)
                task_answer.append(token.text)
                task_result.append('')
                task_adv_options = []
                for i in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
                    # * VB      Verb, base form
                    # * VBD     Verb, past tense
                    # * VBG     Verb, gerund or present participle
                    # * VBN     Verb, past participle
                    # * VBP     Verb, non-3rd person singular present
                    # * VBZ     Verb, 3rd person singular present
                    # * MD      Modal'''
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
    
    # Функция создающая упражнение: выделение фразы, и варианты ответа: названия частей предложения
    def select_memb_groups(self, text, q_words=1):
        task_type = 'select_memb_groups'
        task_text = text
        task_object = []
        task_options = []
        task_answer = []
        task_result = []
        task_description = 'Определите, чем является выделенная фраза'

        # Для задания выбираются рандомные chunk в количестве q_words
        for chunk in self.__nlp(text).noun_chunks:
            task_object.append(chunk)

        # Сохраняем порядок и выбираем случайные элементы. В тексте должно быть больше одного chunk
        if len(task_object) > 1:
            order = {number:task for number,task in enumerate(task_object)}
            task_object = random.sample(task_object, k=min(q_words, len(task_object)))
            order_list = []
            for i in task_object:
                order_list.append(list(order.keys())[list(order.values()).index(i)])
            order_list, task_object = zip(*sorted(zip(order_list, task_object)))
            task_object = list(task_object)

            for chunk in task_object:
                task_text = task_text.replace(chunk.text, '**'+chunk.text+'**')
                task_answer.append(spacy.explain(chunk.root.dep_))
                task_object = chunk.text
                task_options.append('')
                task_result.append('')

            # Возможные варианты ответа формируются из всех описаний части речи из task_answer
            # Если меньше 3-х вариантов ответа, то дозаполняем оставшиеся случайными значениями
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
    
    # Функция создающая упражнение: 3 предложения на выбор, нужно выбрать предложение с правильными формами глагола
    def select_sent_verb(self, text):
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
                # Составляем лист форм слов, не включающий оригинальную форму глагола
                verb_forms = []
                for i in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
                    # * VB      Verb, base form
                    # * VBD     Verb, past tense
                    # * VBG     Verb, gerund or present participle
                    # * VBN     Verb, past participle 
                    # * VBP     Verb, non-3rd person singular present
                    # * VBZ     Verb, 3rd person singular present
                    # * MD      Modal'''
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
    
    # Функция создающая упражнение: 3 предложения на выбор, нужно выбрать из предложения с правильным словом, с син. или ант.
    def select_sent_word(self, text, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1):
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

        # Фильтр, чтобы не появлялись упражнения для предложений, в которых менее 3х токенов, 
        # и в которых нет слова нужной части речи
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
        task_type = 'fill_words_in_the_gaps'
        task_text = text     
        task_object = []     
        task_options = []    
        task_answer = []     
        task_result = []     
        task_description = 'Заполните пропущенное слово (вместе с первой буквой, если была использована подсказка)'
        # проход по токенам с помощью pyinflect
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
    
    # Функция, создающая датасет из всех упражнений для каждого 
    def create_lesson(self, df, start_row=1, q_task=20, 
                      list_of_exercises=[True, True, True, True, True, True, True], q_words=[1, 1, 1, 1, 1, 1, 1]):
        # С точки зрения пользователя, первая строка это 1, а с точки зрения обработки - 0, поэтому вычтем 1
        start_row = start_row-1
        q_task_fact = 0
        lesson_tasks = pd.DataFrame(columns=['row_num', 'raw', 'task_type', 'task_text', 'task_object', 'task_options', 
                                      'task_answer', 'task_result', 'task_description', 'task_total'])

        #Для каждой строки записываем все возможные упражнения
        for i in range(start_row, len(df)):
            mark = 0
            if q_task_fact < q_task:
                row_tasks = pd.DataFrame(columns=['raw', 'task_type', 'task_text', 'task_object', 'task_options', 
                                                  'task_answer', 'task_result', 'task_description', 'task_total'])
                if list_of_exercises[0]:
                    row_tasks.loc[mark] = self.select_word_syn_ant(df.loc[i, 'raw'], q_words=q_words[0])
                    mark += 1
                if list_of_exercises[1]:
                    row_tasks.loc[mark] = self.select_word_adv(df.loc[i, 'raw'])
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
                row_tasks = row_tasks[row_tasks['raw'].isna() == False]
                row_tasks['row_num'] = df.loc[i, 'row_num']
                row_tasks = row_tasks[row_tasks['task_description'].isna() == False]

                # Если хоть какое-то упражнение выгрузилось, то добавляем плюс 1 в счетчик
                if len(row_tasks[row_tasks['task_type'] != 'sent_with_no_exercises']) != 0:
                    q_task_fact += 1
                lesson_tasks = pd.concat([lesson_tasks, row_tasks], ignore_index=True)

        return lesson_tasks
    
    def create_default_lesson(self, df):
        # Перемешиваем все строки, потом сортируем по номеру строки и выбираем первое попавшееся упражнение
        new_df = df[df['task_type'] != 'sent_with_no_exercises'].sample(frac=1).sort_values(by='row_num')
        new_df = new_df.groupby('row_num').agg('first').reset_index()

        df_with_no_exer = df[~df['row_num'].isin(new_df['row_num'].unique())]
        new_df = pd.concat([new_df, df_with_no_exer], ignore_index=True).sort_values('row_num').reset_index(drop=True)

        return new_df
    
    def show_result_table(self, df):
        new_df = df[['task_description', 'task_text', 'task_answer', 'task_result', 'task_total']]
        new_df['task_answer'] = new_df['task_answer'].astype('str')
        new_df['task_result'] = new_df['task_result'].astype('str')
        new_df.columns = ['Тип задания', 'Текст упражнения', 'Правильный ответ', 'Ответ на задание', 'Результат']
        return new_df
    
    def show_result_by_task_type(self, df):
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
            result_mistakes = 'Ошибки были в упражнениях следующих типов: ' + task_with_mistakes + '. В следующий раз попробуй сделать упор именно на такие упражнения'
        
        return result_info + "\n" + result_comment + "\n" + result_mistakes
    
    def display_dataset(self, df):
        new_df = df
        for column in df.columns:
            new_df[column] = new_df[column].astype('str')
        return new_df
    