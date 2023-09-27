import streamlit as st
from exercisegen import ExerciseGen

st.header('Генератор упражнений по английскому языку')

#############################################################################################
# Загрузка текста. 
# Объявляем три опции: загрузка файла, вставка текста, использование стандартного текста
st.subheader('Загрузите текстовый файл или вставьте в поле текст для создания упражнения')

if 'lesson_file' not in st.session_state:
    st.session_state['lesson_file'] = None
if 'lesson_text' not in st.session_state:
    st.session_state['lesson_text'] = None
if 'lesson_default_text' not in st.session_state:
    st.session_state['lesson_default_text'] = None
    
tab1, tab2, tab3 = st.tabs(["Загрузить текстовый файл", "Вставить текст", "Использовать стандартный текст"])
with tab1:
    st.session_state['lesson_file'] = st.file_uploader('nolabel', label_visibility="hidden")
with tab2:
    st.session_state['lesson_text'] = st.text_area('nolabel', label_visibility="hidden")
with tab3:
    st.session_state['lesson_default_text'] = '''Little Red Riding Hood

Charles Perrault

Once upon a time there lived in a certain village a little country girl, the prettiest creature who was ever seen. Her mother was excessively fond of her; and her grandmother doted on her still more. This good woman had a little red riding hood made for her. It suited the girl so extremely well that everybody called her Little Red Riding Hood.
One day her mother, having made some cakes, said to her, "Go, my dear, and see how your grandmother is doing, for I hear she has been very ill. Take her a cake, and this little pot of butter.

Little Red Riding Hood set out immediately to go to her grandmother, who lived in another village.

As she was going through the wood, she met with a wolf, who had a very great mind to eat her up, but he dared not, because of some woodcutters working nearby in the forest. He asked her where she was going. The poor child, who did not know that it was dangerous to stay and talk to a wolf, said to him, "I am going to see my grandmother and carry her a cake and a little pot of butter from my mother."

"Does she live far off?" said the wolf

"Oh I say," answered Little Red Riding Hood; "it is beyond that mill you see there, at the first house in the village."

"Well," said the wolf, "and I'll go and see her too. I'll go this way and go you that, and we shall see who will be there first."

The wolf ran as fast as he could, taking the shortest path, and the little girl took a roundabout way, entertaining herself by gathering nuts, running after butterflies, and gathering bouquets of little flowers. It was not long before the wolf arrived at the old woman's house. He knocked at the door: tap, tap.

"Who's there?"'''
    st.write(st.session_state['lesson_default_text'])

    
# Кнопка генерации упражнений. Проверяем, что именно было загружено, загружаем и обрабатываем данные
if 'generation_clicked' not in st.session_state:
    st.session_state['generation_clicked'] = False
def generation_click_button():
    st.session_state['generation_clicked'] = True
st.button('Загрузить текст упражнений', on_click=generation_click_button)

# Загружаем файл после нажатия кнопки
if st.session_state.generation_clicked:
    if 'dataset' not in st.session_state:
        st.session_state['dataset'] = None
    if st.session_state['lesson_file'] is not None and st.session_state['lesson_text'] != '':
        st.write('Выберите только одну опцию: либо загрузка файла, либо вставка текста в поле')
    elif st.session_state['lesson_file'] is not None:
        with st.spinner('Обработка файла...'):
            if 'ex_gen' not in st.session_state:
                st.session_state['ex_gen'] = ExerciseGen()
            st.session_state['dataset'] = st.session_state['ex_gen'].open_file(st.session_state['lesson_file'])
            st.session_state['dataset'] = st.session_state['ex_gen'].beautify_text(st.session_state['dataset'])
    elif st.session_state['lesson_text'] != '':
        with st.spinner('Обработка загруженного текста...'):
            if 'ex_gen' not in st.session_state:
                st.session_state['ex_gen'] = ExerciseGen()
            st.session_state['dataset'] = st.session_state['ex_gen'].open_text(st.session_state['lesson_text'])
            st.session_state['dataset'] = st.session_state['ex_gen'].beautify_text(st.session_state['dataset'])
    elif st.session_state['lesson_file'] is None and st.session_state['lesson_text'] == '':
        with st.spinner('Обработка стандартного текста...'):
            if 'ex_gen' not in st.session_state:
                st.session_state['ex_gen'] = ExerciseGen()
            st.session_state['dataset'] = st.session_state['ex_gen'].open_text(st.session_state['lesson_default_text'])
            st.session_state['dataset'] = st.session_state['ex_gen'].beautify_text(st.session_state['dataset'])
    else:
        pass

#############################################################################################
    # Блок с настройками
    st.subheader('Настройка упражнений для урока')
    # Ввод строки, начиная с которой будут загружаться упражнения
    if 'start_row' not in st.session_state:
        st.session_state['start_row'] = 1
    # Ввод количества упражнений  
    if 'q_task' not in st.session_state:
        st.session_state['q_task'] = 20
    # Ввод перечня типов упражнений. Всего 7 доступных типов: 
    # 'select_word_syn_ant', 'select_word_adv', 'select_word_verb', 'select_memb_groups', 
    # 'select_sent_verb', 'select_sent_word', 'fill_words_in_the_gaps'
    if 'list_of_exercises' not in st.session_state:
        st.session_state['list_of_exercises'] = [True, True, True, True, True, True, True, True]
    # Ввод количества пропусков и замен в упражнениях
    if 'q_task_exercises' not in st.session_state:
        st.session_state['q_task_exercises'] = [1, 1, 1, 1, 1, 1, 1, 1]
    
    # Ввод строки, начиная с которой будут загружаться упражнения
    st.session_state['start_row'] = st.number_input(label='Введите номер первой строки', 
                                                    min_value = 1, 
                                                    max_value = len(st.session_state['dataset']) - 1,
                                                    value = 1,
                                                    step = 1)
    # Ввод количества упражнений
    st.session_state['q_task'] = st.number_input(label='Введите количество упражнений', 
                                                 min_value = 1, 
                                                 max_value = len(st.session_state['dataset']) - 1,
                                                 value = min(20, len(st.session_state['dataset'])-1),
                                                 step = 1)
    # Ввод перечня типов упражнений
    st.write('Выберите типы упражнений для тестирования')
    st.session_state['list_of_exercises'][0] = st.checkbox(label='Выбор правильного слова', value=True)
    st.session_state['list_of_exercises'][1] = st.checkbox(label='Выбор правильной формы прилагательного', value=True)
    st.session_state['list_of_exercises'][2] = st.checkbox(label='Выбор правильной формы глагола', value=True)
    st.session_state['list_of_exercises'][3] = st.checkbox(label='Выбор предложения с правильным словом', value=True)
    st.session_state['list_of_exercises'][4] = st.checkbox(label='Выбор предложения с нужной формой прилагательного', value=True)
    st.session_state['list_of_exercises'][5] = st.checkbox(label='Выбор предложения с нужной формой глагола', value=True)
    st.session_state['list_of_exercises'][6] = st.checkbox(label='Выбор правильного наименования части речи', value=True)
    st.session_state['list_of_exercises'][7] = st.checkbox(label='Ввод пропущенного слова', value=True)   

    # Ввод количества пропусков и замен в упражнениях
    st.write('Введите максимальное количество пропускаемых или заменяемых слов')        
    st.session_state['q_task_exercises'][0] = st.number_input(label='Выбор правильного слова', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    st.session_state['q_task_exercises'][1] = st.number_input(label='Выбор правильной формы прилагательного', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    st.session_state['q_task_exercises'][2] = st.number_input(label='Выбор правильной формы глагола', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    st.session_state['q_task_exercises'][3] = st.number_input(label='Выбор предложения с правильным словом', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    st.session_state['q_task_exercises'][4] = st.number_input(label='Выбор предложения с нужной формой прилагательного', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    st.session_state['q_task_exercises'][5] = st.number_input(label='Выбор предложения с нужной формой глагола', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    st.session_state['q_task_exercises'][6] = st.number_input(label='Выбор правильного наименования части речи', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    st.session_state['q_task_exercises'][7] = st.number_input(label='Ввод пропущенного слова', 
                                                              min_value = 1, max_value = 10, value = 1, step = 1)
    
    # Кнопка "Применить настройки"
    if 'settings_clicked' not in st.session_state:
        st.session_state['settings_clicked'] = False
    def settings_click_button():
        st.session_state['settings_clicked'] = True
    st.button('Применить настройки', on_click=settings_click_button)
    
    if st.session_state.settings_clicked:
        with st.spinner('Генерация упражнений...'):
    
#############################################################################################

        # Генерация упражнений
        # Создаем упражнения
            if 'lesson_dataset' not in st.session_state:
                st.session_state['lesson_dataset'] = st.session_state['ex_gen'].create_lesson(st.session_state['dataset'], 
                    start_row=st.session_state['start_row'], 
                    q_task=st.session_state['q_task'],
                    list_of_exercises=st.session_state['list_of_exercises'],
                    q_words=st.session_state['q_task_exercises'])
            if 'default_lesson' not in st.session_state:
                st.session_state['default_lesson'] = st.session_state['ex_gen'].create_default_lesson(st.session_state['lesson_dataset'])

            st.subheader('Упражнения по английскому')

            # Выводим упражнения на экран и записываем ответы
            count_tasks = 1
            for i in range(len(st.session_state['default_lesson'])):

                task = st.session_state['default_lesson'].loc[i]    

                # Вывод номера задания для всех предложений, для которых удалось создать упражнение
                if task['task_type'] != 'sent_with_no_exercises':
                    st.write('Задание #'+ str(count_tasks) + ': ' +str(task['task_description']))
                    count_tasks += 1

                # Вывод предложений, для которых не удалось создать упражнение
                if task['task_type'] == 'sent_with_no_exercises':
                    st.write(str(task['task_text']))

                # Вывод предложений с выбором правильного варианта предложения. Окно selectbox с выбором варианта будет только одно
                elif task['task_type'] in ['select_sent_word', 'select_sent_adj', 'select_sent_verb']:
                    task['task_result'][0] = st.selectbox('nolabel', 
                                                          ['–––'] + task['task_options'], 
                                                          label_visibility="hidden",
                                                          key = str(i))

                # Вывод предложений с вводом пропущеного текста. Текстовых полей будет выводится столько, сколько пропущено полей
                elif task['task_type'] == 'fill_words_in_the_gaps':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(str(task['task_text']))
                    with col2:
                        for j in range(len(task['task_answer'])):
                            task['task_result'][j] = st.text_input('nolabel',
                                                                   value='–––', 
                                                                   label_visibility="hidden",
                                                                   key = str(i) + '_' + str(j))

                else:
                    # Вывод предложений с выбором 
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(str(task['task_text']))
                    with col2:
                        for j in range(len(task['task_options'])):
                            option = task['task_options'][j]
                            task['task_result'][j] = st.selectbox('nolabel', 
                                                                  ['–––'] + option, 
                                                                  label_visibility="hidden",
                                                                  key = str(i) + '_' + str(j))
                '---'          

#############################################################################################
            # Кнопка вывода результата теста
            if 'result_clicked' not in st.session_state:
                st.session_state['result_clicked'] = False
            def result_click_button():
                st.session_state['result_clicked'] = True
            st.button('Узнать результат', on_click=result_click_button)

            # Результат прохождения теста
            if st.session_state.result_clicked:
                st.session_state['default_lesson']['task_total'] = (st.session_state['default_lesson']['task_answer'] ==
                                                                    st.session_state['default_lesson']['task_result'])

                st.balloons()
                st.write(st.session_state['ex_gen'].result_interpretation(st.session_state['default_lesson']))
                st.write('Расшифровка результатов в разрезе типов упражнений:')
                st.dataframe(st.session_state['ex_gen'].show_result_by_task_type(st.session_state['default_lesson']))
                st.write('Полная расшифровка результатов прохождения теста:')
                st.dataframe(st.session_state['ex_gen'].show_result_table(st.session_state['default_lesson']))
                st.write('Скачайте файл с расшифровкой, чтобы поделиться результатом:')

                @st.cache_data
                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                csv = convert_df(st.session_state['default_lesson'])

                st.download_button(
                    label="Скачать результат теста в CSV",
                    data=csv,
                    file_name='english_lesson_result.csv',
                    mime='text/csv',
                )
        
# Дальнейший план:

# 0. Сделать так, чтобы при замене пропуском нужного слова заменялся только нужный токен, а не все слова подряд
# 1. Поэтапно блокировать возможность изменений в блоках
# 2. Настроить expander для настроек, либо вывести их на боковую панель
# 3. Дописать функции для упражнений, чтобы во всех можно было выбрать количество пропущенных слов
# 4. Добавить настройку частей речи
# 5. Добавить настройку вывода подсказок
# 6. В задании с пропущенным словом сделать дополнительную подсказку, что это за часть речи
# 7. Добавить страницу ученика (открыта по умолчанию, на ней только загрузка текста, осн.настройки и упражнения) и страницу учителя
# На странице учителя открываются все выбранные предложения (полностью, т.е. ответ заранее известен), и для каждого можно выбрать:
# тип упражнения, часть речи, кол-во пропущенных слов или конкретные слова
# Добавить на страницу учителя возможность сохранить датасет после всех настроек и выгрузить в файл. На блок загрузки добавить опцию
# "Загрузить готовый урок". Это фича, чтобы учитель мог у себя составить урок, и отправить его ученику, а он смог бы его запустить у себя

