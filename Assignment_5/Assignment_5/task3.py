import sys
from blackbox import BlackBox
import random
import binascii

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_path = sys.argv[4]
    random.seed(553)

    bx = BlackBox()
    results = []
    resevoir = []
    n = 0
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file_path, stream_size)

        if _ == 0:
            for user in stream_users:
                resevoir.append(user)
        else:
            for user in stream_users:
                n += 1
                keep_probabily = 100 / (100 + n)

                if random.random() < keep_probabily:
                    to_be_replaced = random.randint(0, 99)
                    resevoir[to_be_replaced] = user

        results.append(((_ + 1) * 100, resevoir[0], resevoir[20], resevoir[40], resevoir[60], resevoir[80]))

    with open(output_file_path, "w") as f:
        f.writelines("seqnum,0_id,20_id,40_id,60_id,80_id")
        f.write("\n")
        for item in results:
            f.writelines(str(item[0]) + "," + str(item[1]) + "," + str(item[2]) + "," + str(item[3]) + "," + str(
                item[4]) + "," + str(item[5]))
            f.write("\n")
