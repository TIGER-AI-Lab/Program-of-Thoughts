
from .tatqa_metric import TaTQAEmAndF1

def test_em_and_f1():

    mode1_test_data = [
        ({'answer_type':'span', 'answer': ['here is, a test'], 'scale':''}, 'here is, a test', '', 1, 1),
        ({'answer_type': 'span', 'answer': ['here is, a test'], 'scale': ''}, 'here is, a test', '', 1, 1),
        ({'answer_type': 'span', 'answer': ['1234.1'], 'scale': 'million'}, '1234.1', 'thousand', 0, 0), # scale mismatch
        ({'answer_type': 'span', 'answer': ['1234.1'], 'scale': 'million'}, '123', 'thousand', 0, 0), # scale mismatch
        ({'answer_type': 'span', 'answer': ['12314.1'], 'scale': 'million'}, '12314.1', 'million', 1, 1),

        ({'answer_type': 'multi-span', 'answer': ['singapore', 'china', 'usa'], 'scale': ''}, ['singapore', 'china', 'usa'], '', 1, 1),
        ({'answer_type': 'multi-span', 'answer': ['singapore', 'china', 'usa'], 'scale': ''}, ['china', 'singapore', 'usa'], '', 1, 1),
        ({'answer_type': 'multi-span', 'answer': ['singapore', 'china', 'usa'], 'scale': ''}, ['china', 'singapore'], '',0, 0.8),

        ({'answer_type': 'arithmetic', 'answer': 123.2, 'scale': 'million'}, 123.2, '', 0, 0), # scale mismatch, f1 = 0
        ({'answer_type': 'arithmetic', 'answer': 123.2, 'scale': 'million'}, 123200000, '', 1, 1), #
        ({'answer_type': 'arithmetic', 'answer': 123.2, 'scale': 'million'}, 123.2, 'thousand', 0, 0), # scale mismatch
        ({'answer_type': 'arithmetic', 'answer': 123.2, 'scale': ''}, 123.2, '', 1, 1),
        ({'answer_type': 'arithmetic', 'answer': 123.22, 'scale': ''}, 123.2, '', 0, 0),
        ({'answer_type': 'arithmetic', 'answer': 123.2, 'scale': ''}, 123.2010, '', 1, 1),
        ({'answer_type': 'count', 'answer': 5, 'scale': ''}, 5, '', 1, 1),
        ({'answer_type': 'arithmetic', 'answer': 22.12, 'scale': 'percent'}, 0.2212, '', 1, 1),
        ({'answer_type': 'arithmetic', 'answer': 22.12, 'scale': 'percent'}, 0.22121, 'percent', 0, 0),
        ({'answer_type': 'arithmetic', 'answer': 22.12, 'scale': 'percent'}, 22.1231, '', 0, 0),
        ({'answer_type': 'arithmetic', 'answer': 22.12, 'scale': 'percent'}, 22.1231, 'percent', 1, 1),
        ({'answer_type': 'span', 'answer': [22.12], 'scale': 'million'}, '22.12', 'million', 1, 1),
        ({'answer_type': 'span', 'answer': [22.12], 'scale': 'million'}, '22.12', '', 0, 0),
        ({'answer_type': 'arithmetic', 'answer': 22.12, 'scale': 'million'}, 'test', '', 0, 0),
        ({'answer_type': 'arithmetic', 'answer': 22.12, 'scale': 'million'}, ["1","2"], '', 0, 0),# span is calcuated by word f1
        ({'answer_type': 'span', 'answer': [22.12], 'scale': 'percent'},"-22.12", '', 0, 0),
        ({'answer_type': 'span', 'answer': [22.12], 'scale': 'percent'},"22.12%", '', 1, 1),
        ({'answer_type': 'span', 'answer': [22.12], 'scale': ''}, "22.12%", '', 0, 0),
        ({'answer_type': 'span', 'answer': [22.12], 'scale': 'million'}, "$22.12", '', 0, 0),
        ({'answer_type': 'arithmetic', 'answer': 22.12, 'scale': 'million'}, "$22.12", '', 0, 0),
        ({'answer_type': 'span', 'answer': ["22.12"], 'scale': 'percent'}, ["-22.12"], '', 0, 0),
        ({'answer_type': 'span', 'answer': ['$1.0 million'], 'scale': ''}, ["['$1.0 million']"], '', 1, 1),

        ({'answer_type': 'span', 'answer': [22.12], 'scale': ''}, "$22.12", '', 1, 1),
        ({'answer_type': 'span', 'answer': [22.12], 'scale': 'percent'}, "22.12%", 'percent', 1, 1),
        ({'answer_type': 'count', 'answer': 5, 'scale': ''}, 'abcd 5', '1', 0, 0),

        ({'answer_type': 'multi-span', 'answer': ['$23,234', '$234.12'], 'scale': ''}, ['234.12', '23,234'], '',
         1, 1),
        ({'answer_type': 'multi-span', 'answer': ['$35,120', '$24,159'], 'scale': ''}, ['$24,159', '$35,120'], '', 1, 1),
        ({'answer_type': 'arithmetic', 'answer': ['34.12'], 'scale': 'percent'}, ['0.3412'], '', 1, 1),
        ({'answer_type': 'span', 'answer': [
            'wages and salaries, social security costs, pension and other costs and share-based payments, see note 10 of the Financial Statements'],
          'scale': ''},
         ['wages and salaries, social security costs, pension and other costs and share - based payments,'], '', 0,
         0.67),

    ]
    metrics = TaTQAEmAndF1()

    for ans, pred, pred_scale, em, f1 in mode1_test_data:
        metrics(ans, pred, pred_scale)
        pred_em, pred_f1, scale_score, op_score = metrics.get_overall_metric(reset=True)
        assert pred_em == em, f'mode2 - pred_em: {pred_em}, em:{em}, pred:{pred}, ans:{ans}'
        assert pred_f1 == f1, f'mode2 - pred_f1: {pred_f1}, f1:{f1}, pred:{pred}, ans:{ans}'


def test_one():
    mode_test_data = [
        ({'answer_type': 'arithmetic', 'answer': ['34.12%'], 'scale': 'percent'}, ['0.3412'], '', 1, 1),
        ({'answer_type': 'arithmetic', 'answer': ['34.12%'], 'scale': ''}, ['0.3412'], '', 1, 1),
    ]
    metrics = TaTQAEmAndF1()
    for ans, pred, pred_scale, em, f1 in mode_test_data:
        metrics(ans, pred, pred_scale)
        pred_em, pred_f1, scale_score, op_score = metrics.get_overall_metric(reset=True)
        assert pred_f1 == f1, f'mode2 - pred_f1: {pred_f1}, f1:{f1}, pred:{pred}, ans:{ans}'
