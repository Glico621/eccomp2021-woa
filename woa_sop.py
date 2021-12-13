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
                a_decrease=0.001,           #変数aの減少値
                logarithmic_spiral=1,       #対数螺旋の係数
    ):
        #各変数の初期化
        self.whale_max = whale_max
        self.a_decrease = a_decrease
        self.logarithmic_spiral = logarithmic_spiral


    #初期化，処理する配列を持ってくる
    def init(self, pop):
        #popの中にある配列の要素数を知りたい
        #print(f'クジラ内pop{pop}')
        self.pop_size = len(pop[0]) - 1    #これ-1いるかいらんか

        #self.best_whale = np.zeros(self.pop_size)
        self._a = 2

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
        self.best_whale = np.array(self.whales[0])

        #変更後の遺伝子を返す用配列
        self.new_whales = []

        #print(pop[num][:-1])
        #print(self.whales)


    #アルゴリズムの処理をここで
    def step(self):
        #print(self.whales)
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
                    new_pos = self.whales[random.randint(0, len(self.whales) -1)]

                new_pos = np.asarray(new_pos)

                
                D = np.linalg.norm(np.multiply(C, new_pos) - pos)
                pos = new_pos = np.multiply(A, D)

            else:
                #旋回
                best_pos = self.best_whale
                #best_pos = np.zeros(self.pop_size)
                D = np.linalg.norm(best_pos - pos)
                L = np.random.uniform(-1, 1, self.pop_size)
                _b = self.logarithmic_spiral
                pos = np.multiply(np.multiply(D, np.exp(_b*L)), np.cos(2.0*np.pi*L)) + best_pos




            
            #四捨五入でもして少数から01に変換する必要がある
            pos = np.where(pos>=0.5, 1, 0)
            #pos = np.where(pos<0.5, pos, 0)
            #print("いかpos")
            #print(pos)

            #!これくっそ問題 →たぶんok
            #計算し終えたposで，whaleを更新
            self.new_whales.append(pos.tolist())

            #最良クジラと比較して，良いほうをbest_whaleに入れる
            #比較方法がわからん
            #ここで計算しないで，ga_sopのupdate()でやるのかな

            #配列最後尾に給付金額を追加
            self.new_whales[count].append(self.pay[count])
            count += 1


        self._a -= self.a_decrease
        if self._a < 0:
            self._a = 0

        print(f'クジラリターン{self.new_whales}')
        #pop（だと思う）を返す
        return self.new_whales





