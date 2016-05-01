####author: yunfeng yuan

import json
import csv
import xlwt
from xlwt import *
import datetime
import os


def main():
    # right_answer question_id --> right_answer
    right_answer = _right_answer(os.path.join('src/', 'annotations.json'))
    # description question_id --> predict_answer
    description = _description(right_answer, os.path.join('src/', 'question_description.json'))
    # predict_answer  question_id --> predict_answer
    predict_answer = _predict_answer(right_answer, os.path.join('predict/', 'new32000_0p8_f_hehe.json'))
    wb = _excel(right_answer, description, predict_answer)
    d = datetime.date.today()
    wb.save(os.path.join('result/', 'result_' + str(d.month) + '_' + str(d.day) + '.xls'))
    _submission(predict_answer, os.path.join('result/', 'submission_'+ str(d.month) + '_' + str(d.day) + '.csv'))


def _right_answer(filename):
    right_answer = {}
    with open(filename) as f:
        data = json.loads(f.read())
        for imgInfo in data['annotations']:
            if imgInfo['answer_type'] == 'yes/no':
                answer = imgInfo['answers'][0]['answer']
                right_answer[imgInfo['question_id']] = answer
    return right_answer


def _description(right_answer, filename):
    description = {}
    right_answer_copy = right_answer.copy()
    with open(filename) as f:
        data = json.loads(f.read())
        for question in data['questions']:
            if question['question_id'] in right_answer_copy:
                description[question['question_id']] = question['question']
    return description


def _predict_answer(right_answer, filename):
    all_predict_answer = {}
    predict_answer = {}
    with open(filename) as f:
        data = json.loads(f.read())
        # make data be a dict     question_id  --> answer
        for i in range(len(data)):
            all_predict_answer[data[i]['question_id']] = data[i]['answer']
        for id in right_answer.keys():
            if id in all_predict_answer:
                predict_answer[id] = all_predict_answer[id]

    return predict_answer


def _excel(right_answer, description, predict_answer):
    global correct
    wb = Workbook()
    sheet = wb.add_sheet('Result')

    sheet.write(0, 0, 'question_id')
    sheet.write(0, 1, 'question')
    sheet.write(0, 2, 'right_answer')
    sheet.write(0, 3, 'predict_answer')
    i = 0
    question_id = []
    for id in right_answer:
        question_id.insert(i, id)
        i += 1
        correct = 0
    for row in range(len(question_id)):
        sheet.write(row + 1, 0, question_id[row])
        sheet.write(row + 1, 1, description[question_id[row]])
        sheet.write(row + 1, 2, right_answer[question_id[row]])
        sheet.write(row + 1, 3, predict_answer[question_id[row]])
        if predict_answer[question_id[row]] != right_answer[question_id[row]]:
            sheet.write(row + 1, 4, 'Wrong')
        else:
            sheet.write(row + 1, 4, 'Correct')
            correct += 1

    style = xlwt.XFStyle()
    pattern = xlwt.Pattern()
    pattern.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern.pattern_fore_colour = xlwt.Style.colour_map['dark_purple']
    style.pattern = pattern

    sheet.write(5, 5, 'Total_yesorno_question_number')
    sheet.write(5, 6, len(question_id), style)
    sheet.write(6, 5, 'CorrectNumber')
    sheet.write(6, 6, correct, style)
    sheet.write(7, 5, 'WrongNumber')
    sheet.write(7, 6, len(question_id) - correct, style)
    sheet.write(8, 5, 'Correct_Rate')
    sheet.write(8, 6, correct / len(question_id), style)

    return wb


def _submission(predict_answer, filename):
    with open(filename, 'w') as csvfile:
        i = 0
        all_question_id = []
        for id in predict_answer.keys():
            all_question_id.insert(i, id)
            i += 1
        fieldnames = ['imageid_questionid', 'multiple_choice_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(all_question_id)):
            question_id = str(all_question_id[i])
            image_id = question_id[0:5]
            full_id = str(image_id + '_' + question_id)
            writer.writerow({'imageid_questionid': full_id, 'multiple_choice_answer': predict_answer[int(question_id)]})

if __name__ == '__main__':
    main()
