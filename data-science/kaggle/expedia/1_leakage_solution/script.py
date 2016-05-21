import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict


def run_solution():
    print('Preparing arrays...')
    with open("../input/train.csv", "r") as f:
        f.readline()
        best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
        best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
        best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
        best_hotel_country = defaultdict(lambda: defaultdict(int))
        popular_hotel_cluster = defaultdict(int)
        total = 0
        line = f.readline().strip()
        while line:
            total += 1

            if total % 10000000 == 0:
                print('Read {} lines...'.format(total))

            arr = line.split(",")
            book_year = int(arr[0][:4])
            user_location_city = arr[5]
            orig_destination_distance = arr[6]
            srch_destination_id = arr[16]
            is_booking = int(arr[18])
            hotel_country = arr[21]
            hotel_market = arr[22]
            hotel_cluster = arr[23]

            append_1 = 3 + 17*is_booking
            append_2 = 1 + 5*is_booking

            if user_location_city != '' and orig_destination_distance != '':
                best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1
            if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
            if srch_destination_id != '':
                best_hotels_search_dest1[srch_destination_id][hotel_cluster] += append_1
            if hotel_country != '':
                best_hotel_country[hotel_country][hotel_cluster] += append_2
            popular_hotel_cluster[hotel_cluster] += 1
            line = f.readline().strip()

    print('Generate submission...')

    with open("../input/test.csv", "r") as f:
        now = datetime.datetime.now()
        path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
        with open(path, "w") as out:
            out.write('id,hotel_cluster\n')
            topclasters = nlargest(5, sorted(
                popular_hotel_cluster.items()), key=itemgetter(1))
            line = f.readline().strip()
            while line:
                total += 1
                if total % 10000000 == 0:
                    print('Write {} lines...'.format(total))

                arr = line.split(",")
                id = arr[0]
                user_location_city = arr[6]
                orig_destination_distance = arr[7]
                srch_destination_id = arr[17]
                hotel_country = arr[20]
                hotel_market = arr[21]

                out.write(str(id) + ',')
                filled = []

                s1 = (user_location_city, orig_destination_distance)
                if s1 in best_hotels_od_ulc:
                    d = best_hotels_od_ulc[s1]
                    topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
                    for i in xrange(len(topitems)):
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
                    for i in xrange(len(topitems)):
                        if topitems[i][0] in filled:
                            continue
                        if len(filled) == 5:
                            break
                        out.write(' ' + topitems[i][0])
                        filled.append(topitems[i][0])
                elif srch_destination_id in best_hotels_search_dest1:
                    d = best_hotels_search_dest1[srch_destination_id]
                    topitems = nlargest(5, d.items(), key=itemgetter(1))
                    for i in xrange(len(topitems)):
                        if topitems[i][0] in filled:
                            continue
                        if len(filled) == 5:
                            break
                        out.write(' ' + topitems[i][0])
                        filled.append(topitems[i][0])

                if hotel_country in best_hotel_country:
                    d = best_hotel_country[hotel_country]
                    topitems = nlargest(5, d.items(), key=itemgetter(1))
                    for i in xrange(len(topitems)):
                        if topitems[i][0] in filled:
                            continue
                        if len(filled) == 5:
                            break
                        out.write(' ' + topitems[i][0])
                        filled.append(topitems[i][0])

                for i in xrange(len(topclasters)):
                    if topclasters[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topclasters[i][0])
                    filled.append(topclasters[i][0])

                out.write("\n")
    print('Completed!')

run_solution()
