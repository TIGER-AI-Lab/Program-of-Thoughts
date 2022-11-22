import json
from time import sleep
from tqdm import tqdm
import os
import openai
from datetime import datetime
from tool import parse_api_result, safe_execute
from eval_tatqa.tatqa_metric import TaTQAEmAndF1
from typing import Dict, Any
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--greedy", default=False, action='store_true')
parser.add_argument("--dry_run", default=False, action='store_true')
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()

def create_reader_request_processed(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then answer a question:\n'
    if example['text']:
        prompt += example['text'].strip() + '\n'
    prompt += example['table'].strip() + '\n'
    question = example['question']
    prompt += f'Quesetion: {question}\n'
    #prompt += 'Answer:'
    return prompt

prompt_8shot = """
Read the following text and table, and then answer a question:
ASSUMPTIONS USED IN STOCK OPTION PRICING MODEL
The fair value of options granted was determined using a variation of a binomial option pricing model that takes into account factors specific to the share incentive plans, such as the vesting period. The following table shows the principal assumptions used in the valuation.
Expected dividend growth is commensurate with BCE’s dividend growth strategy. Expected volatility is based on the historical volatility of BCE’s share price. The risk-free rate used is equal to the yield available on Government of Canada bonds at the date of grant with a term equal to the expected life of the options
— | 2019 | 2018
Weighted average fair value per option granted | $2.34 | $2.13
Weighted average share price | $58 | $57
Weighted average exercise price | $58 | $56
Expected dividend growth | 5% | 5%
Expected volatility | 14% | 12%
Risk-free interest rate | 2% | 2%
Expected life (years) | 4 | 4
Question: How is the fair value of options granted determined?
The answer can be found directly in the text above. So the answer is:
using a variation of a binomial option pricing model that takes into account factors specific to the share incentive plans, such as the vesting period


Read the following text and table, and then answer a question:
7. Employee numbers and costs
The average monthly number of employees (including Executive Directors but excluding third-party contractors) employed by the Group was as follows:
— | 2019 | 2018
— | Number | Number
Customer operations | 370 | 380
Product and technology | 317 | 312
Corporate | 115 | 130
Total | 802 | 822
Question: What are the categories of employees listed in the table?
The answer can be found directly in the table above. So the answer is:
['Customer operations', 'Product and technology', 'Corporate']


Read the following text and table, and then answer a question:
Lines of Credit
The following table summarizes our available lines of credit and committed and uncommitted lines of credit, including the revolving credit facility discussed above, and the amounts available under our accounts receivable securitization programs.
(1) Includes total borrowings under the accounts receivable securitization programs, the revolving credit facility and borrowings under lines of credit available to several subsidiaries.
(2) Of the total available lines of credit, $1,137.4 million were committed as of December 31, 2019.
— | December 31, | —
(In millions) | 2019 | 2018
Used lines of credit (1) | $ 98.9 | $ 232.8
Unused lines of credit | 1,245.2 | 1,135.3
Total available lines of credit(2) | $ 1,344.1 | $ 1,368.1
Quesetion: How much was commited as of December 31, 2019 of total available lines of credit?
The answer can be found directly in the text above. So the answer is:
$1,137.4 million


Read the following text and table, and then answer a question:
Consolidated
The table below presents a summary of our results of operations for fiscal years 2019 and 2018. See Part II, Item 7 of our Annual Report on Form 10-K for the fiscal year ended March 31, 2018, filed with the SEC on May 21, 2018, for Management’s Discussions and Analysis of Financial Condition and Results of Operations for the fiscal year ended April 1, 2017.
REVENUE
Our overall revenue increased $116.8 million in fiscal 2019, compared to fiscal 2018, primarily due to higher demand for our mobile products in support of customers based in China as well as higher demand for our base station products, partially offset by a decrease in revenue due to weakness in marquee smartphone demand experienced by our largest end customer.
We provided our products to our largest end customer (Apple) through sales to multiple contract manufacturers, which in the aggregate accounted for 32% and 36% of total revenue in fiscal years 2019 and 2018, respectively. Huawei accounted for approximately 13% and 8% of our total revenue in fiscal years 2019 and 2018, respectively. These customers primarily purchase RF and Wi-Fi solutions for cellular base stations and a variety of mobile devices, including smartphones, wearables, laptops, tablets and cellular-based applications for the IoT. In May 2019, the U.S. government imposed restrictions on the sales of products to Huawei (see Note 2 of the Notes to the Consolidated Financial Statements set forth in Part II, Item 8 of this report).
International shipments amounted to $2,610.0 million in fiscal 2019 (approximately 84% of revenue) compared to $2,449.1 million in fiscal 2018 (approximately 82% of revenue). Shipments to Asia totaled $2,446.3 million in fiscal 2019 (approximately 79% of revenue) compared to $2,329.3 million in fiscal 2018 (approximately 78% of revenue).
GROSS MARGIN
Gross margin was relatively flat for fiscal 2019 as compared to fiscal 2018, with average selling price erosion offset by favorable changes in product mix.
OPERATING EXPENSES
Research and Development
In fiscal 2019, R&D spending increased $5.4 million, compared to fiscal 2018, primarily due to higher personnel related costs, partially offset by lower product development spend driven by R&D efficiency initiatives.
Selling, General and Administrative
In fiscal 2019, selling, general and administrative expense decreased $51.7 million, or 9.8%, compared to fiscal 2018, primarily due to lower intangible amortization, partially offset by higher personnel related costs.
Other Operating Expense
In fiscal 2019, other operating expense was $52.2 million. In fiscal 2019, we recognized $15.9 million of asset impairment charges (to adjust the carrying value of certain property and equipment to reflect fair value) and $11.6 million of employee termination benefits as a result of restructuring actions (see Note 11 of the Notes to the Consolidated Financial Statements set forth in Part II, Item 8 of this report for information on restructuring actions). In fiscal 2019, we also recorded $18.0 million of start-up costs related to new processes and operations in existing facilities.
In fiscal 2018, other operating expense was $103.8 million. In fiscal 2018, we initiated restructuring actions to improve operating efficiencies, and, as a result of these actions, we recorded approximately $18.3 million of employee termination benefits and adjusted the carrying value of certain held for sale assets located in China and the U.S. to fair market value (resulting in impairment charges totaling approximately $46.3 million). In fiscal 2018, we also recorded integration costs and restructuring costs of $6.2 million and $2.6 million, respectively, associated with the Business Combination, as well as $24.3 million of start-up costs related to new processes and operations in both existing and new facilities.
OPERATING INCOME
Our overall operating income was $216.5 million for fiscal 2019, compared to $70.3 million for fiscal 2018. This increase was primarily due to lower intangible amortization, higher revenue, and lower impairment charges on property and equipment.
— | 2019 | — | 2018 | —
(In thousands, except percentages) | Dollars | % of Revenue | Dollars | % of Revenue
Revenue | $3,090,325 | 100.0% | $2,973,536 | 100.0%
Cost of goods sold | 1,895,142 | 61.3 | 1,826,570 | 61.4
Gross profit | 1,195,183 | 38.7 | 1,146,966 | 38.6
Research and development | 450,482 | 14.6 | 445,103 | 15.0
Selling, general, and administrative | 476,074 | 15.4 | 527,751 | 17.7
Other operating expense | 52,161 | 1.7 | 103,830 | 3.5
Operatingincome | $216,466 | 7.0% | $70,282 | 2.4%
Quesetion: What are the company's respective revenue in 2018 and 2019?
The answer can be found directly in the table above. So the answer is:
['$2,973,536 thousand', '$3,090,325 thousand']


Read the following text and table, and then answer a question:
17. Income Taxes
Income before income taxes for the Company’s domestic and foreign operations was as follows:
— | — | Years Ended June 30, | —
($ in millions) | 2019 | 2018 | 2017
Domestic | $204.2 | $140.3 | $56.0
Foreign | 11.8 | 19.9 | 14.2
Income before income taxes | $216.0 | $160.2 | $70.2
Quesetion: What was the change in Foreign in 2019 from 2018?
The foreign income tax was 11.8 million in 2019 and 19.9 million in 2018. So the answer is:
-8.1 million


Read the following text and table, and then answer a question:
The following table sets forth the breakdown of revenues by category and segment. Travel revenue includes travel publications (Top 20, Website, Newsflash, Travelzoo Network), Getaway vouchers and hotel platform. Local revenue includes Local Deals vouchers and entertainment offers (vouchers and direct bookings) (in thousands).
Revenue by geography is based on the billing address of the advertiser. Long-lived assets attributed to the U.S. and international geographies are based upon the country in which the asset is located or owned.
Year Ended December 31, | — | —
— | 2019 | 2018
Asia Pacific | — | —
Travel | $6,274 | $7,351
Local | 216 | 508
Total Asia Pacific revenues | 6,490 | 7,859
Europe | — | —
Travel | 32,081 | 30,856
Local | 4,817 | 5,293
Total Europe revenues | 36,898 | 36,149
North America | — | —
Travel | 57,863 | 56,145
Local | 10,161 | 11,169
Total North America revenues | 68,024 | 67,314
Consolidated | — | —
Travel | 96,218 | 94,352
Local | 15,194 | 16,970
Total revenues | $111,412 | 111,322
Question: In 2019, how many geographic regions have total revenues of more than $20,000 thousand?
Asia Pacific has total revenues of $6,490 thousand, Europe has total revenues of $36,898 thousand, and North America has total revenues of $68,024 thousand. Europe and North America have total revenues of more than $20,000 thousand. So the answer is:
2


Read the following text and table, and then answer a question:
11 Intangible assets (continued)
(a) Intangible assets
RIGHTS AND LICENCES
Certain licences that NEXTDC possesses have an indefinite useful life and are carried at cost less impairment losses and are subject to impairment review at least annually and whenever there is an indication that it may be impaired.
Other licences that NEXTDC acquires are carried at cost less accumulated amortisation and accumulated impairment losses. Amortisation is recognised on a straight-line basis over the estimated useful life. The estimated useful life and amortisation method are reviewed at the end of each annual reporting period.
INTERNALLY GENERATED SOFTWARE
Internally developed software is capitalised at cost less accumulated amortisation. Amortisation is calculated using the straight-line basis over the asset’s useful economic life which is generally two to three years. Their useful lives and potential impairment are reviewed at the end of each financial year.
SOFTWARE UNDER DEVELOPMENT
Costs incurred in developing products or systems and costs incurred in acquiring software and licenses that will contribute to future period financial benefits through revenue generation and/or cost reduction are capitalised to software and systems. Costs capitalised include external direct costs of materials and services and employee costs.
Assets in the course of construction include only those costs directly attributable to the development phase and are only recognised following completion of technical feasibility and where the Group has an intention and ability to use the asset.
— | Rights and licenses | Internally generated software | Software under development | Total
Movements | $'000 | $'000 | $'000 | $'000
At 30 June 2019 | — | — | — | —
Cost | 13 | 12,961 | 16,284 | 29,259
Accumulated amortisation | - | -5,580 | - | -5,580
Netbook amount | 13 | 7,381 | 16,284 | 23,678
30 June 2018 | — | — | — | —
Opening net book amount at 1 July 2017 | 43 | 442 | 8,053 | 8,538
Additions – externally acquired | 13 | - | 5,253 | 5,266
Additions – internally developed | - | - | 1,256 | 1,256
Amortisation | -43 | -1,746 | - | -1,789
Transfers | - | 7,563 | -7,563 | -
Transfer between classes | - | 744 | - | 744
Disposals | - | -618 | -490 | -1,108
Closing net book amount | 13 | 6,385 | 6,509 | 12,907
At 30 June 2018 | — | — | — | —
Cost | 104 | 9,555 | 6,509 | 16,168
Accumulated amortisation | -91 | -3,170 | - | -3,261
Net book amount | 13 | 6,385 | 6,509 | 12,907
Quesetion: Which year have greater total accumulated amortisation?
The total accumulated amortisation in 2019 is $5,580 thousand and the total accumulated amortisation in 2018 is $3,261 thousand. 2019 has greater total accumulated amortisation. So the answer is:
2019


Read the following text and table, and then answer a question:
Effective Income Tax Rate
A reconciliation of the United States federal statutory income tax rate to our effective income tax rate is as follows:
In 2019 and 2018 we had pre-tax losses of $19,573 and $25,403, respectively, which are available for carry forward to offset future taxable income. We made determinations to provide full valuation allowances for our net deferred tax assets at the end of 2019 and 2018, including NOL carryforwards generated during the years, based on our evaluation of positive and negative evidence, including our history of operating losses and the uncertainty of generating future taxable income that would enable us to realize our deferred tax.
— | Year Ended | Year Ended
— | December 31, 2018 | December 31, 2019
United States federal statutory rate | 21.00% | 21.00%
State taxes, net of federal benefit | 1.99% | -0.01%
Valuation allowance | -21.96% | -24.33%
Cumulative effect of accounting change | — | 2.07%
R&D Credit | 1.34% | 1.53%
Other | -0.38% | -0.27%
Effective income tax rate | 1.99% | -0.01%
Quesetion: What was the 2019 percentage change in pre-tax losses?
The pre-tax losses in 2019 is $19,573 and the pre-tax losses in 2018 is $25,403. The net change in pre-tax losses is -$5,830. The percentage change is -22.95%. So the answer is:
-22.95%
"""


if __name__ == "__main__":
    with open('data/tatqa_dev.json') as f:
        tatqa_dev = json.load(f)
    tatqa_dev = tatqa_dev[args.start:args.end]

    em_and_f1 = TaTQAEmAndF1()
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    if args.greedy:
        filename = f'outputs/tatqa_cot_s{args.start}_e{args.end}_{dt_string}.jsonl'
    else:
        filename = f'outputs/tatqa_cot_sc_{args.start}_e{args.end}_{dt_string}.jsonl'
    writer = open(filename, 'w')

    for example in tqdm(tatqa_dev):
        test_id = example['question_id']
        question = example['question']
        gt_ans = example['answer']
        gt_scale = example['scale']
        gt_derivation = example['derivation']

        full_prompt = prompt_8shot + "\n\n"
        full_prompt += create_reader_request_processed(example)
        if args.dry_run:
            print(full_prompt)
            print('=======================')
            break

        if args.greedy:
            # greedy decoding
            got_result = False
            while not got_result:
                try:
                    result = openai.Completion.create(
                        engine='code-davinci-002',
                        prompt=full_prompt,
                        api_key=os.getenv(args.key),
                        max_tokens=512,
                        temperature=0.0,
                        top_p=1,
                        n=1,
                        stop=['\n\n'],
                        logprobs=1
                    )
                    got_result = True
                except Exception:
                    sleep(3)
        else:
            # self-consistency decoding
            got_result = False
            while not got_result:
                try:
                    result = openai.Completion.create(
                        engine='code-davinci-002',
                        prompt=full_prompt,
                        api_key=os.getenv(args.key),
                        max_tokens=512,
                        temperature=0.5,
                        top_p=1,
                        n=30,
                        stop=['\n\n'],
                        logprobs=1
                    )
                    got_result = True
                except Exception as e:
                    sleep(3)
        # self-consistency decoding or greedy decoding.
        answer_counter = Counter()
        units_counter = Counter()
        codes = parse_api_result(result)
        for r in codes:
            ans = r.split('answer is:')[-1].strip()
            if ans is not None:
                answer_counter.update([str(ans)])

        if len(answer_counter) > 0:
            pred_answer = answer_counter.most_common(1)[0][0]
            if pred_answer.startswith('['):
                try:
                    pred_answer = eval(pred_answer)
                except:
                    pred_answer = pred_answer
        else:
            pred_answer = ''

        pred_scale = ''
        # Furthe Process according to TATQA dataset
        if type(pred_answer) == str:
            pred_answer = [pred_answer]
        if type(pred_answer) == list and type(pred_answer[0]) == str:
            if pred_scale and pred_scale in pred_answer[0]:
                pred_scale = ''
        
        em_and_f1(ground_truth=example, prediction=pred_answer, pred_scale=pred_scale)
        print(em_and_f1.get_overall_metric()[0])
        example.update({'generated': codes, 'pred_answer': pred_answer, 'pred_scale': pred_scale})
        writer.write(json.dumps(example) + '\n')

    print()
    print('em score:', em_and_f1.get_overall_metric()[0])
    writer.close()
