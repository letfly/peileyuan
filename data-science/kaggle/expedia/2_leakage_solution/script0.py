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
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0
    count_empty = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        # cut the word
        arr = line.split(",")
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_1 = 3 + 17*is_booking

        # user_location_city and orig_destination_distance is not null,record to best_hotels_od_ulc{('user_location_city','orig_destination_distance'): defaultdict(<type 'int'>, {'hotel_cluster': booking_weight})}
        if user_location_city != '' and orig_destination_distance != '':
            # best_hotels_od_ulc plus (3+17*is_booking) for loop
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += append_1

        # srch_destination_id hotel_country and hotel_market is not null,record to best_hotels_search_desti{('srch_destination_id', 'hotel_country', 'hotel_market'): defaultdict(<type 'int'>, {'hotel_cluster': booking_weight})}
        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
        else:
            count_empty += 1

        popular_hotel_cluster[hotel_cluster] += append_1

    f.close()
    return best_hotels_od_ulc, best_hotels_search_dest, popular_hotel_cluster


def gen_submission(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster):
    print('Generate submission...')
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    out.write("id,hotel_cluster\n")
    # select top 5 from popular_hotel_cluster
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

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

        # according location_city, orig_destination_distance about user, to decision the hotel_cluster
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

        # if the length about filled isn't above 5,claster in topclasters will be filled into the predicted hotel_cluster
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
