import streamlit as st
import gtts 
from exercisegen import ExerciseGen

st.header('Генератор упражнений по английскому языку')

#############################################################################################
# Загрузка текста. Объявляем три опции: загрузка файла, вставка текста, использование стандартного текста
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
    
    
################################################################
# Блок с настройками
with st.expander('Настройка упражнений для урока'):
    
    st.write('**Выберите перечень предложений для генерации упражнений**')
    col1, col2 = st.columns(2)
    # Ввод строки, начиная с которой будут загружаться упражнения
    with col1:
        if 'start_row' not in st.session_state:
            st.session_state['start_row'] = 1
        st.session_state['start_row'] = st.slider(label='Введите номер первой строки', 
                                                  min_value = 1, max_value = 20, value = 1, step = 1)
    # Ввод количества упражнений
    with col2:
        if 'q_task' not in st.session_state:
            st.session_state['q_task'] = 20
        st.session_state['q_task'] = st.slider(label='Введите количество упражнений', 
                                               min_value = 1, max_value = 50, value = 20, step = 1)
    
    # Ввод перечня типов упражнений
    exercise_types = ['Выбор правильного слова', 
                      'Выбор правильной формы прилагательного',
                      'Выбор правильной формы глагола',
                      'Выбор предложения с правильным словом',
                      'Выбор предложения с нужной формой прилагательного', 
                      'Выбор предложения с нужной формой глагола', 
                      'Выбор правильного наименования части речи',
                      'Ввод пропущенного слова',
                      'Ввод пропущенного текста на основе аудиозаписи', 
                      'Расстановка слов в правильном порядке']
    st.write('**Выберите типы упражнений для тестирования**')
    if 'list_of_exercises' not in st.session_state:
        st.session_state['list_of_exercises'] = [True, True, True, True, True, True, True, True, True, True]
    for i in range(len(exercise_types)):
        st.session_state['list_of_exercises'][i] = st.checkbox(label=exercise_types[i], value=True)

    # Ввод количества пропусков и замен в упражнениях
    st.write('**Введите максимальное количество пропускаемых или заменяемых слов**')
    if 'q_task_exercises' not in st.session_state:
        st.session_state['q_task_exercises'] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(exercise_types)-1):
        st.session_state['q_task_exercises'][i] = st.slider(label=exercise_types[i], min_value = 1, max_value = 10, value = 1, step = 1)

        
################################################################     
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
        if 'ex_gen' not in st.session_state:
            with st.spinner('Обработка файла...'):
                st.session_state['ex_gen'] = ExerciseGen()
                st.session_state['dataset'] = st.session_state['ex_gen'].open_file(st.session_state['lesson_file'])
                st.session_state['dataset'] = st.session_state['ex_gen'].beautify_text(st.session_state['dataset'])
    elif st.session_state['lesson_text'] != '':
        if 'ex_gen' not in st.session_state:
            with st.spinner('Обработка загруженного текста...'):
                st.session_state['ex_gen'] = ExerciseGen()
                st.session_state['dataset'] = st.session_state['ex_gen'].open_text(st.session_state['lesson_text'])
                st.session_state['dataset'] = st.session_state['ex_gen'].beautify_text(st.session_state['dataset'])
    elif st.session_state['lesson_file'] is None and st.session_state['lesson_text'] == '':
        if 'ex_gen' not in st.session_state:
            with st.spinner('Обработка стандартного текста...'):
                st.session_state['ex_gen'] = ExerciseGen()
                st.session_state['dataset'] = st.session_state['ex_gen'].open_text(st.session_state['lesson_default_text'])
                st.session_state['dataset'] = st.session_state['ex_gen'].beautify_text(st.session_state['dataset'])
    else:
        pass
    
    
    #######################################################################################################
    # Генерация упражнений
    if st.session_state['dataset'] is not None:
        with st.spinner('Генерация упражнений...'):
            
            # Создаем упражнения
            if 'lesson_dataset' not in st.session_state:
                st.session_state['lesson_dataset'] = st.session_state['ex_gen'].create_lesson(st.session_state['dataset'], 
                    start_row=st.session_state['start_row'], 
                    q_task=st.session_state['q_task'],
                    list_of_exercises=st.session_state['list_of_exercises'],
                    q_words=st.session_state['q_task_exercises'])
            if 'default_lesson' not in st.session_state:
                st.session_state['default_lesson'] = st.session_state['ex_gen'].create_default_lesson(st.session_state['lesson_dataset'])
                
            # Выводим упражнения на экран и записываем ответы
            st.subheader('Упражнения по английскому')
            count_tasks = 1
            for i in range(len(st.session_state['default_lesson'])):

                task = st.session_state['default_lesson'].loc[i]    

                # Вывод номера задания для всех предложений, для которых удалось создать упражнение
                if task['task_type'] != 'sent_with_no_exercises':
                    st.write('**Задание #'+ str(count_tasks) + ':** ' +str(task['task_description']))
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
                elif task['task_type'] in ['fill_words_in_the_gaps', 'listening_fill_chunks']:
                    col1, col2 = st.columns(2)
                    with col1:
                        if task['task_type'] == 'listening_fill_chunks':
                            audiofile_name = 'audiofile_'+str(i)+'.mp3'
                            if audiofile_name not in st.session_state:
                                audiofile = gtts.gTTS(task['raw'])
                                st.session_state[audiofile_name] = audiofile.save(audiofile_name)
                            st.audio(audiofile_name)
                        st.write(str(task['task_text']))
                    with col2:
                        for j in range(len(task['task_answer'])):
                            task['task_result'][j] = st.text_input('nolabel',
                                                                   value='–––', 
                                                                   label_visibility="hidden",
                                                                   key = str(i) + '_' + str(j))
                
                # Вывод предложений с расстановкой слов в правильном порядке
                elif task['task_type'] == 'set_word_order':
                    task['task_result'][0] = st.multiselect('nolabel',
                                                            options=task['task_text'],
                                                            label_visibility="hidden",
                                                            placeholder='Выберите слово из выпадающего списка',
                                                            key = str(i))
                    
                # Вывод предложений с выбором правильного слова
                else:
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

                
            #######################################################################################################
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
                
                result_info, result_comment, result_mistakes = st.session_state['ex_gen'].result_interpretation(st.session_state['default_lesson'])
                st.write(result_info)
                st.write(result_comment)
                st.write(result_mistakes)
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