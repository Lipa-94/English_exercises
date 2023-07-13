# # Проект: генерация упражнений по английскому языку

# ## Загрузка библиотек
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

import en_core_web_sm
import gensim.downloader as api

# малая модель spacy
nlp = en_core_web_sm.load()

# малая модель glove wiki
# внимание - очень долго скачивает, если она еще не установлена
model = api.load("glove-wiki-gigaword-100")


# ## Загрузка данных и преобразование текста

# Функция загрузки текста с разбивкой на предложения
def open_text(text):
    # Загрузка текста
    paragraphs = text.split("\n")
    dataset = pd.DataFrame({'raw': paragraphs})
    dataset = dataset[dataset['raw'] != '']
    
    # Перенос предложений на новые строки
    nlp = spacy.load("en_core_web_sm")
    rows_list = []
    list_text = dataset['raw'].values
    for i in list_text:
        doc = nlp(i)
        a = [sent.text.strip() for sent in doc.sents]
        for j in a:
            rows_list.append(j)
    df = pd.DataFrame(rows_list, columns=['raw'])
    
    return df

# Функция загрузки текста с разбивкой на предложения
def open_file(file):
    # Загрузка текста
    dataset = pd.read_csv(file, names=['raw'], delimiter="\t")
    
    # Перенос предложений на новые строки
    nlp = spacy.load("en_core_web_sm")
    rows_list = []
    list_text = dataset['raw'].values
    for i in list_text:
        doc = nlp(i)
        a = [sent.text.strip() for sent in doc.sents]
        for j in a:
            rows_list.append(j)
    df = pd.DataFrame(rows_list, columns=['raw'])
    
    return df

# Функция, которая добавляет открывающую или закрывающую кавычку в текст
def quots_func(text):
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
def beautify_text(df):
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
    df['raw'] = df['raw'].apply(quots_func)
    
    # Добавление столбца с индексом
    df = df.reset_index()
    df = df.rename(columns={"index": "row_num"})
    
    return df


# ## Функции, создающие упражнения

# ### Функции преобразования текста для упражнения с выбором пропущенного слова

# Функция создающая упражнение: пропуск слова, и варианты ответа: правильный, синоним, антоним
# Проблема: иногда выводит слишком похожие слова
def select_word_syn_ant(text, pos=['NOUN', 'VERB', 'ADJ', 'ADV'], q_words=1):
    task_type = 'select_word_syn_ant'
    task_text = text
    task_object = []
    task_options = []
    task_answer = []
    task_result = []
    task_description = 'Выберите правильное слово:'
    
    # Для задания выбираются рандомные слова в количестве q_words
    for token in nlp(text):
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
            synonyms = model.similar_by_word(token_new)
            synonyms = [ _[0] for _ in synonyms]
            synonyms = [_.text for _ in nlp(' '.join(synonyms)) if not _.is_stop]
            try:
                if synonyms[0] == token_new:
                    token.append(synonyms[1].title() if token[0].istitle() else synonyms[1])
                else:
                    token.append(synonyms[0].title() if token[0].istitle() else synonyms[0])
            except:
                pass

            # Ищем антонимы
            antonyms = model.most_similar(positive=[token_new, 'bad'], negative=['good'])
            antonyms = [ _[0] for _ in antonyms]
            antonyms = [_.text for _ in nlp(' '.join(antonyms)) if not _.is_stop]
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
            'task_description' : task_description
            }

# Функция создающая упражнение: пропуск прилагательного, и варианты ответа: 3 формы прилагательного
def select_word_adv(text):
    task_text = text
    task_type = 'select_word_adv'
    task_object = []
    task_options = []
    task_answer = []
    task_result = []
    task_description = 'Выберите правильную форму прилагательного:'
    
    # изменение степени прилагательного с помощью pyinflect
    for token in nlp(text):
        # выбираем прилагательные, но только те, для которых есть все 3 формы
        if (token.pos_=='ADJ' and 
            token._.inflect('JJ') != None and 
            token._.inflect('JJR') != None and 
            token._.inflect('JJS') != None):
            task_text = task_text.replace(token.text, '_____')
            task_object.append(token)
            task_answer.append(token)
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
            'task_description' : task_description
            }

# Функция создающая упражнение: пропуск глагола, и варианты ответа: 3 формы глагола
def select_word_verb(text):
    task_type = 'select_verb_form'
    task_text = text
    task_object = []
    task_options = []
    task_answer = []
    task_result = []
    task_description = 'Выберите правильную форму глагола:'
    
    # изменение формы глагола с помощью pyinflect
    for token in nlp(text):
        if (token.pos_=='VERB'):
            task_text = task_text.replace(token.text, '_____')
            task_object.append(token)
            task_answer.append(token)
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
            'task_description' : task_description
            }


# ### Функции преобразования текста для упражнений со структурой предложения

# Функция создающая упражнение: выделение фразы, и варианты ответа: названия частей предложения
def select_memb_groups(text, q_words=1):
    task_type = 'select_memb_main_sec'
    task_text = text
    task_object = []
    task_options = []
    task_answer = []
    task_result = []
    task_description = 'Определите, чем является выделенная фраза?'
    
    # Для задания выбираются рандомные chunk в количестве q_words
    for chunk in nlp(text).noun_chunks:
        task_object.append(chunk)
    
    # Сохраняем порядок и выбираем случайные элементы
    if len(task_object) > 1:
        order = {number:task for number,task in enumerate(task_object)}
        task_object = random.sample(task_object, k=min(q_words, len(task_object)))
        order_list = []
        for i in task_object:
            order_list.append(list(order.keys())[list(order.values()).index(i)])
        order_list, task_object = zip(*sorted(zip(order_list, task_object)))
        task_object = list(task_object)

        for chunk in task_object:
            task_answer.append(spacy.explain(chunk.root.dep_))
            task_options.append('')
            task_result.append('')
        task_object = [chunk.text for chunk in task_object]

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
            'task_description' : task_description
            }


# ### Функции cоздания неправильных предложений и выбор правильного

# Функция создающая упражнение: 3 предложения на выбор, нужно выбрать предложение с правильными формами глагола
def select_sent_verb(text):
    task_type = 'select_sent_verb'
    task_text = text
    task_object = [text]
    task_options = [text]
    task_answer = [text]
    task_result = ['']
    task_description = 'Выберите предложение с правильной формой глагола:'
    
    new_sent_1, new_sent_2 = text, text
    i=5
    count_verbs = 0
    for token in nlp(text):
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
                # * MD      Modal
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
            
    if count_verbs > 0:
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
            'task_description' : task_description
            }

# Функция создающая упражнение: 3 предложения на выбор, нужно выбрать из предложения с правильным словом, с син. или ант.
def select_sent_word(text, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1):
    task_type = 'select_sent_word'
    task_text = text
    task_object = [text]
    task_options = [text]
    task_answer = [text]
    task_result = ['']
    task_description = 'Выберите правильное предложение:'
    
    q_words_fact = 0
    q_all_words = 0
    new_sent_1, new_sent_2 = text, text
    i=5
    for token in nlp(text):
        q_all_words += 1
        if (token.pos_ in pos) and (q_words_fact < q_words):
            m, n = np.random.randint(0, i, 2)
            
            new_word_1 = model.most_similar(token.text.lower(), topn=i)[m][0]
            new_word_2 = model.most_similar(positive = [token.text.lower(), 'bad'],
                                            negative = ['good'],
                                            topn=i)[n][0]

            new_word_1 = new_word_1.title() if token.text.istitle() else new_word_1
            new_word_2 = new_word_2.title() if token.text.istitle() else new_word_2
        
            new_sent_1 = new_sent_1.replace(token.text, new_word_1)
            new_sent_2 = new_sent_2.replace(token.text, new_word_2)
            q_words_fact += 1
    
    # Фильтр, чтобы не появлялись упражнения для предложений, в которых менее 3х токенов, 
    # и в которых нет слова нужной части речи
    if q_words_fact > 0 and q_all_words > 3:
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
            'task_description' : task_description
            }


# ### Пропущенные слова, части речи

# Функция создающая упражнение: нужно ввести с клавиатуры пропущенное слово
# Есть выбор, какие части речи пропускать, есть возможность указать долю пропущенных слов, можно использовать подсказку
def fill_words_in_the_gaps(text, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1, hint=True):
    task_type = 'fill_words_in_the_gaps'
    task_text = text     
    task_object = []     
    task_options = []    
    task_answer = []     
    task_result = []     
    task_description = 'Заполните пропущенное слово'
    # проход по токенам с помощью pyinflect
    count_words = 0
    count_missing_words = 0
    for token in nlp(text):
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
                'task_description' : np.nan
               }
    else:
        return {'raw' : text,
                'task_type' : task_type,
                'task_text' : task_text,
                'task_object' : task_object,
                'task_options' : task_options,
                'task_answer' : task_answer,
                'task_result' : task_result,
                'task_description' : task_description
               }


# ### Пропуск упражнения

# Пропуск упражнения
def sent_with_no_exercises(text):
    return {'raw' : text,
            'task_type' : 'sent_with_no_exercises',
            'task_text' : text,
            'task_object' : np.nan,
            'task_options' : np.nan,
            'task_answer' : np.nan,
            'task_result' : np.nan,
            'task_description' : 'Предложение без упражнения'
            }


# ## Формирование готового датасета со сгенерированными упражнениями

# Функция, создающая датасет из всех упражнений для каждого 
def create_lesson(df, start_row=1, q_task=20):
    # С точки зрения пользователя, первая строка это 1, а с точки зрения обработки - 0, поэтому вычтем 1
    start_row = start_row-1
    q_task_fact = 0
    lesson_tasks = pd.DataFrame(columns=['row_num', 'raw', 'task_type', 'task_text', 'task_object', 'task_options', 
                                  'task_answer', 'task_result', 'task_description'])
    
    #Для каждой строки записываем все возможные упражнения
    for i in range(start_row, len(df)):
        mark = 0
        if q_task_fact < q_task:
            row_tasks = pd.DataFrame(columns=['raw', 'task_type', 'task_text', 'task_object', 'task_options', 
                                              'task_answer', 'task_result', 'task_description'])
            row_tasks.loc[0] = select_word_syn_ant(df.loc[i, 'raw'])
            row_tasks.loc[1] = select_word_adv(df.loc[i, 'raw'])
            row_tasks.loc[2] = select_word_verb(df.loc[i, 'raw'])
            row_tasks.loc[3] = select_memb_groups(df.loc[i, 'raw'])
            row_tasks.loc[4] = select_sent_verb(df.loc[i, 'raw'])
            row_tasks.loc[5] = select_sent_word(df.loc[i, 'raw'])
            row_tasks.loc[6] = fill_words_in_the_gaps(df.loc[i, 'raw'])
            row_tasks.loc[7] = sent_with_no_exercises(df.loc[i, 'raw'])
            row_tasks = row_tasks[row_tasks['raw'].isna() == False]
            row_tasks['row_num'] = df.loc[i, 'row_num']
            row_tasks = row_tasks[row_tasks['task_description'].isna() == False]
            
            # Если хоть какое-то упражнение выгрузилось, то добавляем плюс 1 в счетчик
            if len(row_tasks[row_tasks['task_type'] != 'sent_with_no_exercises']) != 0:
                q_task_fact += 1
            lesson_tasks = pd.concat([lesson_tasks, row_tasks], ignore_index=True)
        
    return lesson_tasks

# Функция, создающая набор упражнений для урока. Выводятся все строки из текста, упражнение выбирается случайно
def create_default_lesson(df):
    # Перемешиваем все строки, потом сортируем по номеру строки и выбираем первое попавшееся упражнение
    new_df = df[df['task_type'] != 'sent_with_no_exercises'].sample(frac=1).sort_values(by='row_num')
    new_df = new_df.groupby('row_num').agg('first').reset_index()
    
    df_with_no_exer = df[~df['row_num'].isin(new_df['row_num'].unique())]
    new_df = pd.concat([new_df, df_with_no_exer], ignore_index=True).sort_values('row_num').reset_index(drop=True)
    
    return new_df

# Функция, которая отображает, упражнения какого типа можно проставить
def get_options(lesson_dataset, row_num):
    return lesson_dataset.loc[lesson_dataset['row_num'] == row_num, 'task_description'].unique()

# Функция, с помощью которой можно заменить один тип упражнения на другой
def change_task_type(lesson, row_num, new_type, pos=['NOUN', 'VERB', 'ADV', 'ADJ'], q_words=1):
    new_task = pd.DataFrame(columns=['raw', 'task_type', 'task_text', 'task_object', 'task_options', 
                                      'task_answer', 'task_result', 'task_description'])
    display()
    if new_type == 'select_word_syn_ant':
        new_task.loc[0] = select_word_syn_ant(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0], pos=pos, q_words=q_words)
    elif new_type == 'select_word_adv':
        new_task.loc[0] = select_word_adv(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0])
    elif new_type == 'select_word_verb':
        new_task.loc[0] = select_word_verb(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0])
    elif new_type == 'select_memb_groups':
        new_task.loc[0] = select_memb_groups(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0], q_words=q_words)
    elif new_type == 'select_sent_verb':
        new_task.loc[0] = select_sent_verb(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0])
    elif new_type == 'select_sent_word':
        new_task.loc[0] = select_sent_word(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0], pos=pos, q_words=q_words)
    elif new_type == 'fill_words_in_the_gaps':
        new_task.loc[0] = fill_words_in_the_gaps(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0], 
                                                 pos=pos, q_words=q_words, hint=True)
    else:
        new_task.loc[0] = sent_with_no_exercises(lesson.loc[lesson['row_num'] == row_num, 'raw'].values[0])
    
    new_task['row_num'] = row_num
    
    lesson = lesson[lesson['row_num'] != row_num]
    lesson = pd.concat([lesson, new_task]).sort_values('row_num').reset_index(drop=True)
    return lesson

# Для проверки через Jupiter Notebook
#df = open_file("Little_Red_Cap_Jacob_and_Wilhelm_Grimm.txt")
#df = beautify_text(df)
# Создали массив всех упражнений
#lesson_dataset = create_lesson(df)
# Выбрали случайные упражнения
#default_lesson = create_default_lesson(lesson_dataset)
#display(default_lesson)
# Вывели все опции упражнения для строки 2
# print(get_options(lesson_dataset, row_num=2))
# Заменили в уроке упражнение 2 на fill_words_in_the_gaps
#lesson = change_task_type(default_lesson, 2, 'fill_words_in_the_gaps', pos='ADV', q_words=3)
#display(lesson)


# ## Визуализация работы модели на streamlit

random.seed(123)

st.header('Генератор упражнений по английскому языку')

# Загрузка текста
st.subheader('Загрузите текстовый файл или вставьте в поле текст для создания упражнения')
lesson_file = None
lesson_text = None
tab1, tab2 = st.tabs(["Загрузить текстовый файл", "Вставить текст"])
with tab1:
    lesson_file = st.file_uploader('nolabel', label_visibility="hidden")
with tab2:
    st.subheader('Вставьте в поле текст для создания упражнения')
    lesson_text = st.text_area('nolabel', label_visibility="hidden")

# Если файл загружен, обрабатываем датафрейм
save_button = False
dataset = None
if lesson_file is not None and lesson_text != '':
    st.write('Выберите только одну опцию: либо загрузка файла, либо вставка текста в поле')
elif lesson_file is not None:
    dataset = open_file(lesson_file)
    dataset = beautify_text(dataset)
    save_button = st.button('Сгенерировать упражнения')
elif lesson_text is not None:
    dataset = open_text(lesson_text)
    dataset = beautify_text(dataset)
    save_button = st.button('Сгенерировать упражнения')
else:
    pass

# Будущий блок с настройками


# Генерация упражнений
if save_button:
    st.dataframe(dataset)
    q_task = 20
    start_row = 1
    lesson_dataset = create_lesson(dataset, start_row=start_row, q_task=q_task)
    default_lesson = create_default_lesson(lesson_dataset)
    
    st.subheader('Упражнения по английскому')
    
    for i in range(len(default_lesson)):
        task = default_lesson.loc[i]
        st.write('Задание #'+ str(i) + ': ' +str(task['task_description']))
        
        if task['task_type'] == 'sent_with_no_exercises':
            st.write(str(task['task_text']))
        elif task['task_type'] in ['select_sent_verb', 'select_sent_word']:
            task['task_result'] = st.selectbox('nolabel', 
                                               ['–––'] + task['task_options'], 
                                               label_visibility="hidden",
                                               key = i)
            if task['task_result'] == '–––' or task['task_result'] == []:
                pass
            elif task['task_result'] == task['task_answer']:
                st.success('', icon="✅")
            else:
                st.error('', icon="😟")
        elif task['task_type'] == 'fill_words_in_the_gaps':
            col1, col2 = st.columns(2)
            with col1:
                st.write(str(task['task_text']))
            with col2:
                task['task_result'] = st.text_input(label='Введите пропущенное слово (вместе с первой буквой, если была использована подсказка)', value='---', key=i)
            if task['task_result'] == '–––' or task['task_result'] == [] or task['task_result'] == ['–––']:
                pass
            elif task['task_result'] == task['task_answer']:
                st.success('', icon="✅")
            else:
                st.error('', icon="😟")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write(str(task['task_text']))
            with col2:
                #st.write(str(task['task_options']))
                for j in range(len(task['task_options'])):
                    option = task['task_options'][j]
                    task['task_result'][j] = st.selectbox('nolabel', 
                                                          ['–––'] + option, 
                                                          label_visibility="hidden",
                                                          key = ((j+1)*len(default_lesson) + (i+1)))
                    if task['task_result'][j] == '–––':
                        pass
                    elif task['task_result'][j] == task['task_answer'][j]:
                        st.success('', icon="✅")
                    else:
                        st.error('', icon="😟")
        '---'  

# Дальнейший план:

# Добавить скрываемый блок настроек. В нем: настройка кол-ва упражнений, настройка первой строки, с которой идут упражнения, и типы упр.
# Добавить подсчет результата с интерпретацией, какие упражнения хуже идут. + Подробная таблица. + Файл с расшифровкой (чтобы поделиться)

# Добавить страницу ученика (открыта по умолчанию, на ней только загрузка текста, осн.настройки и упражнения) и страницу учителя
# На странице учителя открываются все выбранные предложения, и для каждого можно выбрать:
# тип упражнения, часть речи, кол-во слов или конкретные слова
# Добавить на страницу учителя возможность сохранить датасет после всех настроек и выгрузить в файл. На блок загрузки добавить опцию
# "Загрузить готовый урок". Это фича, чтобы учитель мог у себя составить урок, и отправить его ученику, а он смог бы его запустить у себя