#woa

#参考までに
#qiita：https://qiita.com/shinjikato/items/afd754a4dfca7efc4cd1
#元本の論文：https://www.sciencedirect.com/science/article/abs/pii/S0965997816300163

"""
処理の流れ

・入力データ，クジラの個体数を初期化
・a, A, C, I,pを初期化
・各探索エージェントの適応度を計算
・X* = 現在の最高の探索エージェント

while (it < Maxiter)
    for each search agent
        if (p < 0.5)
            if (|A| < 1)
                獲物に近づく
            else if (|A| ≥ 1)
                獲物を探す
            end
        else if (p ≥ 0.5)
            回る
        end
    end

・各探索エージェントの適応度を計算
・より良い解がある場合，X*を更新する
・a, A, C, I,pの更新

・X*を返す
"""

import math
import random

import numpy as np




class WOA():
    def __init__(self,
                whale_max=10,               #クジラ頭数
                a_decrease=0.4,           #変数aの減少値
                logarithmic_spiral=1,       #対数螺旋の係数
    ):
        #各変数の初期化
        self.whale_max = whale_max
        self.a_decrease = a_decrease
        self.logarithmic_spiral = logarithmic_spiral
        self._a = 2

        self.past_whales = []

    #初期化，処理する配列を持ってくる
    #def init(self, pop):
        #popの中にある配列の要素数を知りたい
        #print(f'クジラ内pop{pop}')


        #print(pop[num][:-1])
        #print(self.whales)


    #アルゴリズムの処理をここで
    def step(self, pop, hof):
        #print(self.whales)
        self.pop_size = len(pop[0]) - 1    #これ-1いるかいらんか

        #self.best_whale = np.zeros(self.pop_size)

        #最後の要素（値段）を除いた01要素だけ持ってくる
        #self.whales：popの遺伝子を格納
        #self.pay：popの給付金額部分（配列末尾）を格納
        self.whales = []
        self.pay = []
        for num in range(len(pop)):
            prov = pop[num][:-1]
            pay = pop[num][-1]
            self.whales.append(prov)
            self.pay.append(pay)

        #print(f'pay:{self.pay}')

        #!最良を持ってきたい
        best = []
        self.bests = []
        if len(hof)==0:
            self.best_whale = np.array(self.whales[0])
        else:
            best = hof[0][:-1]
            self.best_whale = np.array(best)
            
            for h in hof:
                best = h[:-1]
                self.bests.append(np.array(best))
                self.past_whales.append(np.array(best))

        self.bests += self.whales

        #変更後の遺伝子を返す用配列
        self.new_whales = []


        count = 0
        #クジラ集団の配列から，一つずつ取り出して処理させる
        for whale in self.whales:
            #print(whale)
            pos = whale
            #pos = np.array(whale)
            #print(pos)
            #print(pos)
            #01乱数によって分岐
            if random.random() < 0.5:
                r1 = np.random.rand(self.pop_size)   #要素数分の乱数を生成する，要素数間違ってそう
                r2 = np.random.rand(self.pop_size)

                A = (2.0 * np.multiply(self._a, r1)) - self._a
                C = 2.0 * r2

                #print(f'A:{A}')
                #print(f'C:{C}')

                if np.linalg.norm(A) < 1:       #np.linalg.norm():行列ノルム（距離）を計算
                    #獲物に近づく
                    #目標を最良クジラ（=獲物）に定める
                    new_pos = self.best_whale
                else:
                    #獲物を探す
                    #目標は，ランダムのクジラ
                    #new_pos = self.whales[random.randint(0, len(self.whales) -1)]
                    if len(self.past_whales)==0:
                        new_pos = self.whales[random.randint(0, len(self.whales) -1)]
                    else:
                        #new_pos = self.bests[random.randint(0, len(self.bests) -1)]
                        new_pos = self.past_whales[random.randint(0, len(self.past_whales) -1)]

                new_pos = np.asarray(new_pos)


                D = np.linalg.norm(np.multiply(C, new_pos) - pos)
                pos = new_pos - np.multiply(A, D)

            else:
                #旋回
                best_pos = self.best_whale
                #best_pos = np.zeros(self.pop_size)
                D = np.linalg.norm(best_pos - pos)
                L = np.random.uniform(-1, 1, self.pop_size)
                _b = self.logarithmic_spiral
                pos = np.multiply(np.multiply(D, np.exp(_b*L)), np.cos(2.0*np.pi*L)) + best_pos





            #四捨五入でもして少数から01に変換する必要がある
            pos = np.where(pos>=2, 1, 0)
            #pos = np.where(pos<0.5, pos, 0)
            #print("いかpos")
            #print(pos)

            #計算し終えたposで，whaleを更新
            self.new_whales.append(pos.tolist())


            self.past_whales.append(pos.tolist())


            #配列最後尾に給付金額を追加
            self.new_whales[count].append(self.pay[count])
            count += 1



        #条件に適応させていく
        #まずは，必ず1にする必要があるものから
        true_list = [0,9,10,17,38,40,42,43] # これは必ず1を立てる

        for whale in self.new_whales:
            print(f'whale{whale}')
            for index in true_list:
                if whale[index] == 0:
                    whale[index] = 1
                    print("やあ")

        #次に，優先順位的な問題
        # それぞれ第一優先はtruelistで追加済み
        #家族構成
        family_second = [4]
        family_third = [3]
        family_fourth = [1,2,5,6,7,8]
        #雇用形態
        employment_second = [40,41]
        employment_third = [39]
        #企業規模
        scale_second = [44]
        scale_third = [45]
        scale_fourth = [46]

        #調整していく
        #1を立てるだけでなく，1を消す選択肢も持ちたい
        for whale in self.new_whales:
            #家族構成
                #パターン１：優先順位4で1があった場合，優先順位2,3も1を立てる
                #パターン２：                        優先順位3,4に0を立てる
                #パターン３：                        優先順位2,3,4に0を立てる
            rand01 = random.random()
            for fourth in family_fourth:
                if whale[fourth]==1:
                    for second,third in zip(family_second, family_third):
                        #パターン１
                        if rand01 < 0.2:
                            whale[second] = 1
                            whale[third] = 1
                        #パターン２
                        elif rand01 < 0.6:
                            whale[third] = 0
                            whale[fourth] = 0
                            #for four in family_fourth:
                                #whale[four] = 0
                        #パターン３
                        else:
                            whale[second] = 0
                            whale[third] = 0
                            whale[fourth] = 0
                            #for four in family_fourth:
                            #    whale[four] = 0
                #パターン１：優先順位3で1があった場合，優先順位2も1を立てる
                #パターン２：                        優先順位3,4に0
            rand01 = random.random()
            for third in family_third:
                if whale[third]==1:
                    if rand01 < 0.3:
                        for second in family_second:
                            whale[second] = 1
                    else:
                        whale[third] = 0
                        for fourth in family_fourth:
                            whale[fourth] = 0

            #雇用形態
                #パターン１：優先順位3で1があった場合，優先順位2も1を立てる
                #パターン２：                        優先順位2,3に0
            rand01 = random.random()
            for third in employment_third:
                if whale[third]==1:
                    if rand01 < 0.3:
                        for second in employment_second:
                            whale[second] = 1
                    else:
                        whale[third] = 0
                        for second in employment_second:
                            whale[second] = 0

            #企業規模
                #パターン１：優先順位4で1があった場合，優先順位2,3も1を立てる
                #パターン２：                        優先順位3,4に0を立てる
                #パターン３：                        優先順位2,3,4に0を立てる
            rand01 = random.random()
            for fourth in scale_fourth:
                if whale[fourth]==1:
                    for second,third in zip(scale_second, scale_third):
                        #パターン１
                        if rand01 < 0.2:
                            whale[second] = 1
                            whale[third] = 1
                        #パターン２
                        elif rand01 < 0.6:
                            whale[third] = 0
                            whale[fourth] = 0
                            #for four in family_fourth:
                                #whale[four] = 0
                        #パターン３
                        else:
                            whale[second] = 0
                            whale[third] = 0
                            whale[fourth] = 0
                            #for four in family_fourth:
                            #    whale[four] = 0
                #パターン１：優先順位3で1があった場合，優先順位2も1を立てる
                #パターン２：                        優先順位3,4に0
            rand01 = random.random()
            for third in scale_third:
                if whale[third]==1:
                    if rand01 < 0.3:
                        for second in scale_second:
                            whale[second] = 1
                    else:
                        whale[third] = 0
                        for fourth in scale_fourth:
                            whale[fourth] = 0
        
        """
        for whale in self.new_whales:
            #家族構成
            #優先順位4で1があった場合，優先順位2,3も1を立てる
            for fourth in family_fourth:
                if whale[fourth]==1:
                    for second,third in zip(family_second, family_third):
                        whale[second] = 1
                        whale[third] = 1
            #優先順位3で1があった場合，優先順位2も1を立てる
            for third in family_third:
                if whale[third]==1:
                    for second in family_second:
                        whale[second] = 1
            
            #雇用形態
            #優先順位3で1があった場合，優先順位2も1を立てる
            for third in employment_third:
                if whale[third]==1:
                    for second in employment_second:
                        whale[second] = 1
                        
            #企業規模
            #優先順位4で1があった場合，優先順位2,3も1を立てる
            for fourth in scale_fourth:
                if whale[fourth]==1:
                    for second,third in zip(scale_second, scale_third):
                        whale[second] = 1
                        whale[third] = 1
            #優先順位3で1があった場合，優先順位2も1を立てる
            for third in scale_third:
                if whale[third]==1:
                    for second in scale_second:
                        whale[second] = 1
        """

        #self.past_whales += self.new_whales
        
        self._a -= self.a_decrease
        if self._a < 0:
            self._a = 0

        #print(f'クジラリターン{self.new_whales}')
        #pop（だと思う）を返す
        return self.new_whales





