## Program of Thoughts
This is code repository for the paper "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks"





## Few-shot Results

1. GSM8K
- Number of Test Examples: 1318
- Output: outputs/gsm8K_s0_e-1_11_17_10_20.jsonl
- EM Score: 0.716

- Output: outputs/gsm8K_sc_s0_e-1_11_08_21_14.jsonl
- EM Score: 0.799

2. AQuA
- Number of Test Examples: 253
- Output: outputs/aqua_s0_e-1_11_06_18_38.jsonl
- EM Score: 0.541

- Output: aqua_sc_s0_e-1_11_07_20_49.jsonl
- EM Score: 0.582

3. SVAMP
- Number of Test Examples: 1000
- Output: outputs/svamp_s0_e-1_11_06_21_11.jsonl
- EM Score: 0.835

- Output: outputs/svamp_sc_s0_e-1_11_08_14_02.jsonl
- EM Score: 0.882

4. TabWMP
- Number of Test Examples: 7861
- Output: outputs/tabmwp_s0_e-1_11_06_22_55.jsonl
- EM Score: 0.732

- Output: outputs/tabmwp_sc_s0_e-1_11_08_18_21.jsonl
- EM Score: 0.818

5. FinQA
- Number of Test Examples: 1147 
- Ouptut: outputs/finqa_s0_e-1_11_09_13_15.jsonl
- EM Score: 0.623

- Output: outputs/finqa_sc_s0_e-1_11_09_13_00.jsonl
- EM SCore: 0.651


6. ConvFinQA
- Number of Test Examples: 421 
- Ouptut: outputs/convfinqa_s0_e-1_11_12_01_38.jsonl
- EM Score: 0.665

- Output: outputs/convfinqa_sc_s0_e-1_11_12_02_27.jsonl
- EM SCore: 0.714

7. TATQA
- Number of Test Examples: 1668 
- Output: outputs/tatqa_8shot_11_06_19_53.json
- EM Score: 0.689

- Output: outputs/tatqa_8shot_11_06_19_53.json
- EM Score: 0.702


## Zero-shot Results


1. GSM8K
- Number of Test Examples: 1318
- Output: outputs/gsm8K_zs_s0_e-1_11_19_09_55.jsonl
- EM Score: 0.569

2. AQuA
- Number of Test Examples: 253
- Output: outputs/aqua_zs_s0_e-1_11_19_11_56.jsonl
- EM Score: 0.438
```
python compute_score.py --inputs aqua_zs_s0_e-1_11_19_11_56.jsonl --relaxed
```

3. SVAMP
- Number of Test Examples: 1000
- Output: outputs/svamp_zs_s0_e-1_11_18_20_12.jsonl
- EM Score: 0.708

4. MultiArith
- Number of Test Examples: 600
- Output: outputs/multiarith_zs_s0_e-1_11_19_20_12.jsonl
- EM Score: 0.922

5. TabMWP
- Output: outputs/tabmwp_zs_s0_e-1_11_19_20_01.jsonl
- EM Score: 0.646

