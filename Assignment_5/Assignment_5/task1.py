import sys
from blackbox import BlackBox
import random
import binascii

def myhashs(s):
    result = []
    s = int(binascii.hexlify(s.encode('utf8')), 16)
    hash_function_list = [(1999, 2023, 111317, 69997), (2000, 2001, 111317, 69997), (2022, 2024, 111317, 69997)]
    
    for f in hash_function_list:
        a = f[0]
        b = f[1]
        p = f[2]
        m = f[3]
        result.append(((a * s + b) % p) % m)

    return result

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_path = sys.argv[4]

    bx = BlackBox()
    bit_array = [0] * 69997
    results = []
    prev_users = set()
    
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file_path, stream_size)
        FP = 0
        TN = 0

        for user in stream_users:
            hashed_values = myhashs(user)
            flag_seen = True

            for hashed_value in hashed_values:
                if bit_array[hashed_value] == 0:
                    TN += 1
                    flag_seen = False
                    break

            if flag_seen == True:
                if user not in prev_users:
                    FP += 1
            else:
                for hashed_value in hashed_values:
                    if bit_array[hashed_value] == 0:
                        bit_array[hashed_value] = 1

            prev_users.add(user)

        FPR = FP / (FP + TN)
        results.append((_, FPR))

    with open(output_file_path, "w") as f:
        f.writelines("Time,FPR")
        f.write("\n")
        for item in results:
            f.writelines(str(item[0]) + "," + str(item[1]))
            f.write("\n")


