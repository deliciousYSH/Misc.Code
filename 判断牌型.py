from operator import attrgetter
import random


# 初始化牌堆
def make_cards(poker):
    color = ["♥", "♦", "♣", "♠"]
    num = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  
    for f in color:
        for n in num:
            poker.append(poker_cards(f, n))
    return poker


# 洗牌
def shuffle_cards(poker):
    for id in range(52):
        idd = random.randint(0, 51)
        poker[id], poker[idd] = poker[idd], poker[id]
    return poker


# 发牌
def deal_cards(poker, player):
    for i in range(5):
        player.append(poker.pop(0))
    return player


# 判断顺子
def shunzi(player):
    count_gap = 0
    for i in range(len(player) - 1):
        if player[i + 1].num == player[i].num:
            return False
        count_gap = count_gap + player[i + 1].num - player[i].num - 1
    if count_gap == 0:
        return True
    return False


# 判断条子
def tiaozi(player):
    temp = [0] * 15
    for i in range(len(player)):
        temp[player[i].num] = temp[player[i].num] + 1
    if max(temp) == 4:
        print("四条")
        return True
    elif max(temp) == 3:
        if temp.count(0) == 13:
            print("满堂红")
            return True
        elif temp.count(0) == 12:
            print("三条")
            return True
    elif max(temp) == 2:
        if temp.count(0) == 11:
            print("一对")
            return True
        elif temp.count(0) == 12:
            print("两对")
            return True
    return False


# 打印牌面
def print_cards(player):
    # 排序
    cmpfun = attrgetter("num")
    cflag = 0
    player.sort(key=cmpfun)
    for card in player:
         print(card.color, card.num)

    for card in player:
        if card.color == player[0].color:
            cflag = cflag + 1
    # 判断同花和同花顺
    if cflag == 5:
        if shunzi(player):
            print("同花顺")
        else:
            print("同花")
    # 判断顺子
    elif cflag != 5:
        if shunzi(player):
            print("顺子")

        # 判断条子
        elif tiaozi(player):
            pass
        else:
            print("正常牌")


class poker_cards:

    def __init__(self, color, num):
        self.color = color  # 花色 红桃，黑桃，梅花，方片
        self.num = num  # 点数



poker = []
player = []
make_cards(poker)
shuffle_cards(poker)
deal_cards(poker, player)
print_cards(player)
