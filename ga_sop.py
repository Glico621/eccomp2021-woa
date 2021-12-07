# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# ECコンペ2021 単目的部門の実行サンプル
# 作成者は後藤裕介(芝浦工業大学)です．お問い合わせは y-goto@shibaura-it.ac.jp までお願いします．

# ### 概要
# GAで近似解を導出するpythonプログラムです．
# 最終世代の最良解をcsvとして書き出します．
# シミュレーションプログラムに渡す引数の設定，戻り値の受取り方，とりあえず動く実装として参考になさってください．
# 
# ### 実行方法と注意
# プログラムのあるディレクトリで以下のように実行してください．
# python example_sop.py
# このとき，必ず実行用の変数の設定をご自分の環境に合わせて確認してください．
# 1つの解の評価に MacBook Air (M1, 2020)の環境で20秒程度かかります．
# 1つの子プロセスの展開で実行時に150MB程度のメモリを消費しますので，子プロセスの展開数は
# メモリとCPUコア数とを確認されてから設定してください．
# 
# ### 動作環境
# 以下の環境で動作確認をしています．
# 外部のライブラリとしては[DEAP](https://github.com/deap/deap)を使っています．
# - on macOS Big Sur 11.1
# - deap: 1.3.1
# - multiprocess: 0.70.12.2
# - numpy: 1.20.2
# - pandas: 1.2.4
# - python: 3.9.2
# subprocessの処理でエラーが出る際には，pythonのバージョンを3.7以上に上げることを試してみてください．
import platform
import random
import subprocess
import datetime
import math

import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from example_mop import SIM_PATH

### 実行用の変数の設定
# - N_PROC: 子プロセスの展開数．
# - OUT_DIR: パレートフロントのcsvを書き出すディレクトリ．当初の設定ではカレントディレクトリを指定しています．
# - EID: パレートフロントのcsvの拡張子の前の部分． "p001" とすると，p001.csv として保存します．
# - FID: 目的関数のID．F_1では "[1]" F_2では "[2]" のように指定してください．
# - CITY: 実行する都市名． naha：沖縄県那覇市，hakodate: 北海道函館市．　
# - SEEDS: 実行時の乱数シードのリスト．""で囲って定義してください．
N_PROC = 5
OUT_DIR = "./"
EID = "ga_result"
FID = "[2]"
CITY = "hakodate"
# 単目的部門では， FID "[1]" CITY "naha"　， FID "[2]" CITY "hakodate" の2通りの指定を行えばよいです．
SEEDS = "[123,42,256]"

### GAの設定
# - SEED：GAの遺伝的操作の際の乱数シード．シミュレーションにわたす乱数シードとは異なる点に注意．
# - N_IND：個体数
# - N_GEN：世代数
# - N_ATTR：支給対象を決める部分の遺伝子長．コーディングのしかたによって変更はありえます．
# - N_PAY: 支給金額を決める部分の遺伝子長．例えば，給付金額の調整を細かく行う際には変更が必要．
# - S_TOUR: トーナメントサイズ
# - P_CROSS_1：交叉確率（交叉を行うかどうか決定する確率）
# - P_CROSS_2：交叉確率（一様交叉を行うときに，その遺伝子座が交叉する確率）
# - P_MUTATION：各遺伝子座が突然変異する確率
# - N_HOF: 記録用に保持する(上位n個の)最良個体数
SEED = 42
N_IND = 3
N_GEN = 3
N_ATTR = 47
N_PAY = 16
S_TOUR = 3
P_CROSS_1 = 0.5
P_CROSS_2 = 0.5
P_MUTATION = 0.025
N_HOF = 20

# シミュレータのパス
SIM_PATH = platform.system() + "/syn_pop.py"

#給付優先順位
priority_order_list  = (
    (
        (0,),
        (4,),
        (3,),
        (1,2,5,6,7,8,)
    ),
    (
        (9,10,),
        (11,12,13,14,15,16,)
    ),
    (
        (17,), 
        (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,)
    ),
    (
        (38,),
        (40,41,),
        (39,),
    ),
    (
        (42,),(43,),(44,),(45,),(46,),
    ),
)


## 関数群の定義
# - gene2pay: コーディングした遺伝子から，設計変数へと変換する
# - ret_fitness: 子プロセスが完了することを待って，適応度を返す
# - evaluation: 個体の評価を行う．
# - decode_hof: 最良個体を支援制度（クエリ， 金額）にデコードする支援制度（クエリ， 金額）にデコードする
# - create_valid_pop: 支給対象の制約条件を満たす初期個体を生成する
def gene2pay(gene): 
    ### コーディングした遺伝子から，設計変数へと変換する関数
    # クエリ q は pandas.DataFrame.query の形式で書く形です．
    # シミュレーションプログラムでは制約条件を満たしているかの判定を渡されたクエリの文字列から
    # 行っていますので，スペースの入れ方をここでなされているように書いてください．
    #
    # 引数：
    #   gene: 個体の遺伝子
    # 戻り値：
    #   q: 給付金の対象を決めるクエリ
    #   pay: 給付金額（単位：万円）
    q = ''
    
    family_type_val = [0, 1, 2, 3, 4, 50, 60, 70, 80]
    family_type = [family_type_val[j] for i,j in zip(range(0, 9), range(9)) if gene[i] == 1]
    family_type = ",".join(map(str, family_type))
    q = q + 'family_type_id == [' + family_type + ']'

    role_household_type_val = [0, 1, 10, 11, 20, 21, 30, 31]
    role_household_type = [role_household_type_val[j] for i,j in zip(range(9, 17), range(8)) if gene[i] == 1]
    role_household_type = ",".join(map(str, role_household_type))
    q = q + ' and role_household_type_id == [' + role_household_type + ']'

    industry_type_val = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    industry_type = [industry_type_val[j] for i,j in zip(range(17, 38), range(21)) if gene[i] == 1]
    industry_type = ",".join(map(str, industry_type))
    q = q + ' and industry_type_id == [' + industry_type + ']'

    employment_type_val = [-1, 10, 20 ,30]
    employment_type = [employment_type_val[j] for i,j in zip(range(38, 42), range(4)) if gene[i] == 1]
    employment_type = ",".join(map(str, employment_type))
    q = q + ' and employment_type_id == [' + employment_type + ']'        

    company_size_val = [-1, 5, 10 ,100, 1000]
    company_size = [company_size_val[j] for i,j in zip(range(42, 47), range(5)) if gene[i] == 1]
    company_size = ",".join(map(str, company_size))
    q = q + ' and company_size_id == [' + company_size + ']'
    #万円単位で
    pay = gene[N_ATTR]
    
    """
    pay = 0
    
    for i in range(47, 47 + N_PAY):
        pay += gene[i]
    """

    

    return q, pay

def gene2pay4human(gene): 
    ### コーディングした遺伝子から，設計変数へと変換する関数
    # クエリ q は pandas.DataFrame.query の形式で書く形です．
    # シミュレーションプログラムでは制約条件を満たしているかの判定を渡されたクエリの文字列から
    # 行っていますので，スペースの入れ方をここでなされているように書いてください．
    #
    # 引数：
    #   gene: 個体の遺伝子
    # 戻り値：
    #   q: 給付金の対象を決めるクエリ
    #   pay: 給付金額（単位：万円）
    q = ''
    #family_type_val = [0, 1, 2, 3, 4, 50, 60, 70, 80]
    family_type_val = [
        "単独世帯", 
        "夫婦のみ世帯", 
        "夫婦と子供世帯", 
        "男親と子供", 
        "女親と子供", 
        "男親と両親 (夫の親)", 
        "夫婦とひとり親 (夫の親)", 
        "夫婦・子供と両親 (夫の親)", 
        "夫婦・子供とひとり親 (夫の親)"
    ]
    family_type = [family_type_val[j] for i,j in zip(range(0, 9), range(9)) if gene[i] == 1]
    family_type = ",".join(map(str, family_type))
    #family_type = ", ".join(family_type)
    q = q + '家族類型 A_1 == [' + family_type + ']'

    #role_household_type_val = [0, 1, 10, 11, 20, 21, 30, 31]
    role_household_type_val = [
        "単独世帯 (男性)", 
        "単独世帯 (女性)", 
        "夫・男親", 
        "妻・女親", 
        "子供 (男性)", 
        "子供 (女性)", 
        "親 (男性)", 
        "親 (女性)"
    ]
    role_household_type = [role_household_type_val[j] for i,j in zip(range(9, 17), range(8)) if gene[i] == 1]
    role_household_type = ",".join(map(str, role_household_type))
    q = q + ' \nand 世帯内役割 A_2 == [' + role_household_type + ']'

    #industry_type_val = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    industry_type_val = [
        "非就業者", 
        "A 農業，林業", 
        "B 漁業", 
        "C 鉱業，採石業，砂利採取業", 
        "D 建設業", 
        "E 製造業", 
        "F 電気・ガス・熱供給・水道業", 
        "G 情報通信業", 
        "H 運輸業，郵便業", 
        "I 卸売業，小売業", 
        "J 金融業，保険業", 
        "K 不動産業，物品賃貸業", 
        "L 学術研究，専門・技術サービス業", 
        "M 宿泊業，飲食サービス業", 
        "N 生活関連サービス業，娯楽業", 
        "O 教育，学習支援業", 
        "P 医療，福祉", 
        "Q 複合サービス事業", 
        "R サービス業 (他に分類されないもの)", 
        "S 公務 (他に分類されるものを除く)", 
        "T 分類不能の産業"
    ]
    industry_type = [industry_type_val[j] for i,j in zip(range(17, 38), range(21)) if gene[i] == 1]
    industry_type = ",".join(map(str, industry_type))
    q = q + '\nand 産業分類 A_3 == [' + industry_type + ']'

    #employment_type_val = [-1, 10, 20 ,30]
    employment_type_val = [
        "非就業者", 
        "一般労働者", 
        "短時間労働者", 
        "臨時労働者"
    ]
    employment_type = [employment_type_val[j] for i,j in zip(range(38, 42), range(4)) if gene[i] == 1]
    employment_type = ",".join(map(str, employment_type))
    q = q + '\nand 雇用形態 A_4 == [' + employment_type + ']'        

    #company_size_val = [-1, 5, 10 ,100, 1000]
    company_size_val = [
        "非就業者", 
        "5~9人", 
        "10~99人", 
        "100~999人", 
        "1000人以上"
    ]
    company_size = [company_size_val[j] for i,j in zip(range(42, 47), range(5)) if gene[i] == 1]
    company_size = ",".join(map(str, company_size))
    q = q + '\nand 企業規模 == [' + company_size + ']'

    pay = gene[N_ATTR]

    return q, pay
    
def ret_fitness(p):
    ### 子プロセスが完了することを待って，目的関数値などを返す関数
    # 引数：
    #   p: 子プロセス
    # 戻り値：
    #   Fの目的関数値，Fの各条件での目的関数値,解が制約条件を満たすか（T/F），
    #   解の金額面の余裕（マイナスの場合には制約を満たしていない）
    
    a, err = p.communicate(timeout=10_000)
    # 正常に子プロセスが終了しないときは，目的関数値を1_000にしておく -> 次は選ばれないように
    if p.returncode != 0:
        print("sim failed %d %s %s" % (p.returncode, a, err))
        return 1_000, [1_000], False, [0]
    else:
        a_split = eval(a)
        # 実行
        #print("a_split")
        #print(a_split)
        if a_split[0] == None:
            print('実行不可能')
            return 1_000, a_split[1], a_split[2], a_split[3]
        else:
            print('F : '+str(a_split[0]))
            return float(a_split[0]), a_split[1], a_split[2], a_split[3]

def evaluation(pop):
    ### 個体の評価を行う関数
    # 1個体の評価に時間がかかるため，並行して実行しています．
    # 
    # 引数：
    #   pop: 個体の集合
    # 戻り値：
    #   pop: 評価値を計算した個体の集合
    
    # 各個体の評価値と実行可能かどうかをリストに入れていく
    f_list = []
    is_feasible_list = []

    # 1回あたりの実行に時間がかかるため，子プロセスを生成して，並行して実行する
    # 個体群をN_PROC個を単位として，バッチに分ける．
    # batch_list：バッチを要素とするlist
    # ind_list: 1バッチを構成する個体のlist
    n_ind = len(pop)
    batch_list, ind_list = [], []
    for i in range(n_ind):
        ind_list.append(i)
        # 以下の条件でバッチにまとめる
        # (1)バッチで処理する子プロセスが満たされたとき
        # (2)(1)でないが，最後の個体となったとき
        if (i + 1) % N_PROC == 0 or i == n_ind - 1:
            batch_list.append(ind_list)
            ind_list = []
            
    # バッチごとに処理を進めていく
    # job_list: 実行するコマンドを要素とするlist
    # procs：subprocessに展開するためのlist
    for ind_list in batch_list:
        job_list, procs = [], []
        for i in ind_list:
            ind = pop[i]
            q, pay = gene2pay(ind)
            #!書き換え python3 → pyhton
            cmd = ["python", SIM_PATH, str(q), str(pay), str(FID), str(CITY), str(SEEDS)]

            #print('CMD :'+' '.join(cmd))
            job_list.append(cmd)
        procs = [subprocess.Popen(job, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for job in job_list]

        for i in range(len(ind_list)):
            # avg: 目的関数値
            # vals: 各条件での実行値のlist(valsを平均したものがavg)
            # judge: 解が制約条件（条件の優先関係）を満たしているか？
            # slacks: 金額面の制約の違反量（正の場合にはまだ余裕がある．負の場合には違反している量） 
            # vals, slacksはここでは使っていませんが，アルゴリズムやパラメータの検討時に参考になると思われます
            avg, vals, judge, slacks = ret_fitness(procs[i])
            f_list = f_list + [avg]
            is_feasible_list = is_feasible_list + [judge]

    for ind, f, j in zip(pop, f_list, is_feasible_list):
        # 目的関数値を各個体に割り当てていく．
        # このときに，解が金額の制約以外で，実行可能でないときには，ペナルティとして，目的関数値を1000とする
        if j == False:
            ind.fitness.values = 1_000,
        else:
            ind.fitness.values = f,

    return pop

def decode_hof(hof):
    # パレートフロントの個体を支援制度（クエリ， 金額）にデコードする
    # 引数
    #   hof: パレートフロントの個体
    # 戻り値：
    #   支援制度（クエリ， 金額）のDataFrame
    q_and_pay = []
    for h in hof:
        q, p = gene2pay(h)
        f = h.fitness.values[0]
        gene = h
        q_and_pay.append([q, p, f, gene])
    return pd.DataFrame(q_and_pay, columns=['query', 'payment', 'f', 'gene'])

def is_feasible(gene):
        for attribute_type in priority_order_list:
            constrations = True
            next_constrations = True
            for priority_group in attribute_type:
                for index in priority_group:
                    #参照してるpriority_groupに給付対象とならない属性が含まれている場合、次以降のグループはすべて0でなければいけない
                    if gene[index] == 0 and next_constrations:
                        next_constrations = False
                    elif constrations == False and gene[index] == 1:
                        #優先順位を満たしてない場合はFalseを返す
                        return False
                    else:
                        pass
                constrations = next_constrations
        """
        benefit_amount = 0
        for i in range(N_ATTR, N_ATTR + N_PAY):
            benefit_amount += gene[i]
        """
        #q, benefit_amount = gene2pay(gene)
        #benefit_upper_limit = calc_benefit_upper_limit(gene)
        #if benefit_upper_limit < benefit_amount:
        
        if calc_benefit_upper_limit(gene) < gene[N_ATTR]:
            return False
        else:
            #print("上限 : {}, 給付額 : {}".format(calc_benefit_upper_limit(gene), sum(gene[-N_PAY:])))
            return True

def create_valid_pop_uniformly():
    valid_pop = []
    true_list = [0,9,10,17,38,40,42,43] # これは必ず1を立てる
    while len(valid_pop) < N_IND:
        tmp = []
        # - 最低の支給（単身で無職）範囲の確定
        # - それ以外の部分は 0.5 の確率で 1 を割当
        for j in range(N_ATTR+N_PAY):
            if j in true_list:
                tmp.append(1)
            elif random.random() < 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
        if is_feasible(tmp):
            valid_pop.append(tmp)
            print('✅')
        else:
            print('⚠️',end='')
    return valid_pop

def create_valid_pop_with_bias():
    ### 支給対象の定義において制約を満たす個体(群)を返す
    # 戻り値：
    #   個体群のリスト（2次元）
    
    true_list = [0,9,10,17,38,40,42,43] # これは必ず1を立てる

    valid_pop = []
    for i in range(N_IND):
        attribute_part = [None for i in range(N_ATTR)]
        for attribute_type in priority_order_list:
            constrations = True
            next_constrations = True
            for priority_group in attribute_type:
                for index in priority_group:
                    if index in true_list:
                        attribute_part[index] = 1
                    elif constrations and random.random() < 0.5:
                        attribute_part[index] = 1
                    else:
                        attribute_part[index] = 0
                        next_constrations = False
                constrations = next_constrations
        """
        #対象者一人当たりの給付額上限を計算
        #print(attribute_part + [0 for i in range(N_PAY)])
        benefit_upper_limit = calc_benefit_upper_limit(attribute_part + [0 for i in range(N_PAY)])
        #上限の範囲で給付額を決定
        benefit_amount = random.randint(0, int(benefit_upper_limit))
        benefit_part = ([1 for i in range(benefit_amount)] + [0 for i in range(N_PAY)])[:N_PAY]
        random.shuffle(benefit_part)
        gene = attribute_part + benefit_part
        """
        benefit_amount = random.random()*calc_benefit_upper_limit(attribute_part+[0])
        gene = attribute_part + [benefit_amount]
        valid_pop.append(gene)
    return valid_pop



def to_feasible(pop):
    for ind in range(len(pop)):
        #優先順位が高い属性についての遺伝子情報を優先的に維持し、優先順位が低い遺伝子は制約条件を満たすように改変する。
        for attribute_type_group in priority_order_list:
            #print('attribute_type_group')
            #print(attribute_type_group)
            constrations = True
            next_constrations = True
            for priority_group in attribute_type_group:
                #print('priority_group')
                #print(priority_group)
                for index in priority_group:
                    #現在のpriority_groupより、優先順位が高い属性で給付対象から漏れている属性がああれば、
                    if pop[ind][index] == 1 and not constrations:
                        pop[ind][index] = 0
                        del pop[ind].fitness.values
                    elif pop[ind][index] == 0 and next_constrations:
                        next_constrations = False
                    else:
                        pass
                constrations = next_constrations
        benefit_upper_limit = calc_benefit_upper_limit(pop[ind])
        diff = sum(pop[ind][-N_PAY:]) - benefit_upper_limit
        #給付額が上限を超えていたら、下げる
        if 0 < diff:
            #給付額パートで1がたってるところインデックスのリストをつくる
            true_position_list = [i for i in range(N_ATTR, N_ATTR+N_PAY) if pop[ind][i] == 1]
            #ランダムにどこかの1を0にかえる
            for i in random.choices(true_position_list, k=math.ceil(diff)):
                pop[ind][i] = 0
                del pop[ind].fitness.values
    #print('全部実行可能化できたかな？')
    #print([["❌", "👍"][int(is_feasible(i))] for i in pop])
    return pop

def calc_benefit_upper_limit(gene):
    city_df_list = [pd.read_csv('data/csv/' + CITY + '_0' + str(i) + '.csv') for i in range(1,4)]
    q, pay = gene2pay(gene)
    total_pop = len(city_df_list[0])
    benefit_target_pop = [ len(i.query(q)) for i in city_df_list ]
    #人工個票によって給付対象人数が異なり、給付額の上限も違ってくる
    #対象人数が最も多い人口個票を参考に上限を決定する
    #ここは円単位の計算
    benefit_upper_limit = (total_pop*20000)/max(benefit_target_pop)
    #対象者一人当たりの給付金額の上限を❗️「万円単位」❗️で返す
    return benefit_upper_limit/10000.0
    
def main():
    ### メインルーチン
    # GAはDEAPを使って実装する
    # 詳細は https://deap.readthedocs.io/en/master/index.html
    # 遺伝子：0 or 1で生成（ランダムに生成．生成/割当のしかたは改善の余地あり）
    # 交叉：一様交叉
    # 突然変異：ビット反転
    # 選択：トーナメント選択
    
    #スクリプトを途中終了したとき警告がでて嫌なので、残ってたら消しておく
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    # 初期の個体は支給対象の定義においては制約を満たす個体で始める（なお，金額面は満たすとは限らない）
    random.seed(SEED)
    valid_pop = create_valid_pop_with_bias()
    #amount_part_pop = create_amount_part_pop()

    def initPopulation(pcls, ind_init, file):    
        return pcls(ind_init(c) for c in file)
    
    def hybrid_crossover(ind1, ind2, indpb):
        """Executes a uniform crossover that modify in place the two
        :term:`sequence` individuals. The attributes are swapped according to the
        *indpb* probability.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param indpb: Independent probability for each attribute to be exchanged.
        :returns: A tuple of two individuals.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        #　参考 : https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py#L73
        size = min(len(ind1), len(ind2))
        for i in range(size):
            #最後の要素は実数値なので特別に処理する
            if i == size-1:
                
                #(とりあえず平均をとることにする)
                avrg = (ind1[i]+ind2[i])/2.0
                ind1[i] = avrg
                ind2[i] = avrg
                
                
                """
                #差をとって、差二倍の範囲でランダムに増減させる。
                diff = abs(ind1[i] - ind2[i])
                
                upper_limit =  ind1[i] + diff
                lower_limit = ind1[i] - diff 
                if lower_limit < 0:
                    lower_limit = 0
                ind1[i] = random.uniform(lower_limit, upper_limit)
                
                upper_limit =  ind2[i] + diff
                lower_limit = ind2[i] - diff 
                if lower_limit < 0:
                    lower_limit = 0
                ind2[i] = random.uniform(lower_limit, upper_limit)
                """
                
                """
                #二点間の任意の点を乱数で決定
                new_val_1 = random.uniform(ind1[i], ind2[i])
                new_val_2 = random.uniform(ind1[i], ind2[i])
                ind1[i] = new_val_1
                ind2[i] = new_val_2
                """
                
                """
                #評価値の比に基づいて内分点を決定し、次世代の値を決定する
                fitness_val_sum = ind1.fitness.values[0] + ind2.fitness.values[0]
                internal_division = (ind1.fitness.values[0]/fitness_val_sum)
                ind1[i] = internal_division*ind1[i] + (1.0-internal_division)*ind2[i]
                ind2[i] = (1.0-internal_division)*ind1[i] + internal_division*ind2[i]
                """
                
            elif random.random() < indpb:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
        
    def hybrid_mutate(individual, indpb):
        """Flip the value of the attributes of the input individual and return the
        mutant. The *individual* is expected to be a :term:`sequence` and the values of the
        attributes shall stay valid after the ``not`` operator is called on them.
        The *indpb* argument is the probability of each attribute to be
        flipped. This mutation is usually applied on boolean individuals.
        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be flipped.
        :returns: A tuple of one individual.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        # 参考 : https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py#L124
        for i in range(len(individual)-1):
            if random.random() < indpb:
                individual[i] = type(individual[i])(not individual[i])
                
        #完全なランダムだと範囲が大きすぎるので、とりあえず予算範囲内でランダム
        if random.random() < indpb:
            individual[-1] = random.random() * calc_benefit_upper_limit(individual)
        return individual,
    toolbox.register("population_byhand", initPopulation, list, creator.Individual, valid_pop)
    #toolbox.register("mate", tools.cxUniform)
    toolbox.register("mate", hybrid_crossover)
    #toolbox.register("mutate", tools.mutFlipBit, indpb=P_MUTATION)
    toolbox.register("mutate", hybrid_mutate, indpb=P_MUTATION)
    toolbox.register("select", tools.selTournament, tournsize=S_TOUR)
    
    # 世代数の1/5ごとと，最後の世代において，以下のアーカイブ（スナップショット）を記録しておく
    # - pop_archive: 個体情報のアーカイブ
    # - paretof_archive: パレートフロントのアーカイブ
    pop_archive = []
    paretof_archive = []    
    
    # 個体集合の作成
    pop = toolbox.population_byhand()
    pop_archive.append((0, pop[:]))
    # 個体の評価
    pop = evaluation(pop)
    print([i.fitness.values for i in pop])
    print([i for i in pop if 1000.0 <= i.fitness.values[0]])
    
    # ログ関係
    stats = tools.Statistics()
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "avg", "std", "min", "max"
    record = stats.compile([ind.fitness.values[0] for ind in pop])
    logbook.record(gen=0, evals=len(pop), **record)
    hof = tools.HallOfFame(maxsize=N_HOF)
    
    # 進化のサイクルを回す
    #for g in []:
    for g in range(1, N_GEN + 1):
        print('第'+str(g) + '世代' + ' : ',end='')
        print(datetime.datetime.now()-start_time,end='')
        print(' 経過...')
        # 子の世代の選択と複製
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSS_1:
                toolbox.mate(child1, child2, P_CROSS_2)
                del child1.fitness.values
                del child2.fitness.values
        # 突然変異
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        """
        print('変換前 : ',end='')
        print([["❌", "👍"][int(is_feasible(i))] for i in offspring])
        print([i.fitness.values for i in offspring])
        #offspring = to_feasible(offspring)
        print('変換後 : ',end='')
        print([["❌", "👍"][int(is_feasible(i))] for i in offspring])
        """
        #print([i.fitness.values for i in offspring])
        # 子の世代で無効な適応度（delされたもの）をもつ個体を対象として評価を行う
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = evaluation(invalid_ind)
        # invalid_ind = evaluation(offspring)
        print([i.fitness.values for i in offspring])
        # 子の世代を次の個体集合へ置き換える
        pop[:] = offspring
        
        record = stats.compile([ind.fitness.values for ind in pop])
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        hof.update(pop)
        
    
    # 次回の実行のため，削除しておく
    del creator.FitnessMin
    del creator.Individual
    
    return logbook, hof
# -


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("Start:"+str(start_time))
    # 進化計算の実行
    logbook, hof = main() # logbookはこのサンプルでは利用していない

    # 最良個体の出力
    #print(hof)
    df_hof_final = decode_hof(hof)
    #print(df_hof_final)
    df_hof_final.drop_duplicates(keep='first', subset=['query', 'payment'])
    df_hof_final.to_csv(OUT_DIR + EID + '_p.csv')

    print("最優良解:{}".format(df_hof_final.at[0,'f']))
    end_time = datetime.datetime.now()
    print("End : "+str(end_time))
    print("Delta : "+str(end_time - start_time))


