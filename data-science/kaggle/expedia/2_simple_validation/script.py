import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

# validation ###############
validate = 1  # 1 - validation, 0 - submission
N0 = 2        # total number of parts
N1 = 1        # number of part
#--------------------------

def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    hits = defaultdict(int)
    tp = defaultdict(float)

    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1
        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))
        if line == '':
            break
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

    f.close()
    ###########################
    if validate == 1:
        print('Validation...')
        f = open("../input/train.csv", "r")
    else:
        print('Generate submission...')
        f = open("../input/test.csv", "r")
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f.readline()
    total = 0
    totalv = 0
    leak = 0
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1
        if total % 10000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        if validate == 1:
            book_month = int(arr[0][5:7])
            user_location_city = arr[5]
            orig_destination_distance = arr[6]
            user_id = int(arr[7])
            srch_destination_id = arr[16]
            is_booking = int(arr[18])
            hotel_country = arr[21]
            hotel_market = arr[22]
            hotel_cluster = arr[23]
            id = 0
            if user_id % N0 != N1:
                continue
            if is_booking == 0:
                continue
        else:
            id = arr[0]
            user_location_city = arr[6]
            orig_destination_distance = arr[7]
            user_id = int(arr[8])
            srch_destination_id = arr[17]
            hotel_country = arr[20]
            hotel_market = arr[21]
            is_booking = 1

        totalv += 1
        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)

        if s1 in best_hotels_od_ulc:
            leak +=1
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) == 5:
                    break
                if topitems[i][0] in filled:
                    continue
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                    if topitems[i][0]==hotel_cluster:
                        hits[len(filled)] +=1

        s2 = (srch_destination_id, hotel_country, hotel_market)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) == 5:
                    break
                if topitems[i][0] in filled:
                    continue
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                    if topitems[i][0]==hotel_cluster:
                        hits[len(filled)] +=1

        if srch_destination_id in best_hotels_search_dest1:
            d = best_hotels_search_dest1[srch_destination_id]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) == 5:
                    break
                if topitems[i][0] in filled:
                    continue
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                    if topitems[i][0]==hotel_cluster:
                        hits[len(filled)] +=1


        if hotel_country in best_hotel_country:
            d = best_hotel_country[hotel_country]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if len(filled) == 5:
                    break
                if topitems[i][0] in filled:
                    continue
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                if validate == 1:
                    if topitems[i][0]==hotel_cluster:
                        hits[len(filled)] +=1


        for i in range(len(topclasters)):
            if len(filled) == 5:
                    break
            if topclasters[i][0] in filled:
                continue
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])
            if validate == 1:
                if topclasters[i][0]==hotel_cluster:
                    hits[len(filled)] +=1


        out.write("\n")
    out.close()
    print('Completed!')
    # validation >>>
    scores = 0.0
    classified = 0
    if validate == 1:
        for jj in range(1,6):
            scores +=  hits[jj]*1.0/jj
            tp[jj] = hits[jj]*100.0/totalv
            classified += hits[jj]
        misclassified = totalv-classified
        miscp = misclassified*100.0/totalv
        print("")
        print(" validation")
        print("----------------------------------------------------------------")
        print("position %8d %8d %8d %8d %8d %8d+" % (1,2,3,4,5,6))
        print("hits     %8d %8d %8d %8d %8d %8d " % (hits[1],hits[2],hits[3],hits[4],hits[5],misclassified))
        print("hits[%%]  %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f " % (tp[1],tp[2],tp[3],tp[4],tp[5],miscp))
        print("----------------------------------------------------------------")
        print("MAP@5 = %8.4f " % (scores*1.0/totalv))
        print("leakage = %6.2f%%" % (leak*100.0/totalv))
    # <<< validation
    if validate == 0:
        print("leakage = %6.2f%%" % (leak*100.0/total))

run_solution()
