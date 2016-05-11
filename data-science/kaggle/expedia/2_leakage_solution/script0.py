# coding: utf-8
__author__ = 'letfly: https://kaggle.com/letfly'

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict


def prepare_arrays_match():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    print best_hotels_od_ulc
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    print best_hotels_search_dest
    popular_hotel_cluster = defaultdict(int)
    print popular_hotel_cluster
    total = 0
    count_empty = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        print line
        total += 1

        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        # 首先我们进行切词
        arr = line.split(",")
        # 读取第6个单词
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_1 = 3 + 17*is_booking
        print append_1

        # 如果user_location_city和orig_estination_distance都不为空,纪录{('用户所在城市','用户搜索时与宾馆距离'): defaultdict(<type 'int'>, {'hotel_cluster':订购权重})}
        if user_location_city != '' and orig_destination_distance != '':
            # 通过循环best_hotels_od_ulc加(3+17*is_booking)
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += append_1
            print best_hotels_od_ulc

        # 如果srch_destination_id、hotel_country和hotel_market都不为空,纪录{('进行搜索的酒店所在地','旅馆乡村','旅馆市场'): defaultdict(<type 'int'>, {'hotel_cluster': 订购权重})
        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
            print best_hotels_search_dest
        else:
            count_empty += 1

        popular_hotel_cluster[hotel_cluster] += append_1
        print popular_hotel_cluster
        if total == 10:
            break

    f.close()
    print('Empty: ', count_empty)
    return best_hotels_od_ulc, best_hotels_search_dest, popular_hotel_cluster


def gen_submission(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster):
    print('Generate submission...')
    print best_hotels_od_ulc
    print best_hotels_search_dest
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))
    print topclasters

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        # 首先根据用户所在城市和用户搜索时与宾馆的距离，判断宾馆集群
        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                print filled

        s2 = (srch_destination_id, hotel_country, hotel_market)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        # 如果filled没有超过5个，将把topclasters中的claster填充到预测的hotel_cluster中
        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])

        out.write("\n")
    out.close()
    print('Completed!')


best_hotels_od_ulc, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
gen_submission(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster)
