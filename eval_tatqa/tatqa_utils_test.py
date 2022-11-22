import pytest
from .tatqa_utils import *


def test_extract_first_num_from_text():
    text = '2.3 million'
    assert extract_one_num_from_str(text) == 2.3
    text = '-2.3 million'
    assert extract_one_num_from_str(text) == -2.3
    text = '205 million'
    assert extract_one_num_from_str(text) == 205
    text = '-1,210 million'
    assert extract_one_num_from_str(text) == -1210


def test_to_num():
    text = '2.3 million'
    assert to_number(text) == 2300000
    text = '-2.3 thousand'
    assert to_number(text) == -2300
    text = '205 billion'
    assert to_number(text) == 205000000000
    text = '-1,210 million'
    assert to_number(text) == -1210000000



def test_ws_tokenize():
    text = '2.3   million'
    assert ws_tokenize(text) == ['2.3', 'million']
    text = '2.3 \nmillion'
    assert ws_tokenize(text) == ['2.3', 'million']
    text = '2.3\n\tmillion'
    assert ws_tokenize(text) == ['2.3', 'million']

def test_normalize_answer():
    assert normalize_answer('-134.12') == '-134.12'
    assert normalize_answer('134.12') == '134.12'
    assert normalize_answer('(134.12)') == '-134.12'
    assert normalize_answer('18.3%') == '0.183'



def test_is_num():
    assert is_number('$124')
