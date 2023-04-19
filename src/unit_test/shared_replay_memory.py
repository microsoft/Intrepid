import random
import multiprocessing as mp

def square_list(n, results):
    """
    function to square a given list
    """
    while True:
        i = random.randint(0, 4)
        results[n][i] = random.random()

        # print result Array
        if i == 0:
            print("Result(in process p1) 0: " + str(sum(results[0])))
            print("Result(in process p1) 1: " + str(sum(results[1])))


if __name__ == "__main__":

    # creating Array of int data type with space for 4 integers
    results = []
    for i in range(0, 2):
        results.append(mp.Array('f', range(5)))

    # creating new process
    p1 = mp.Process(target=square_list, args=(0, results))
    p1.start()

    p2 = mp.Process(target=square_list, args=(1, results))
    p2.start()

    # wait until process is finished
    p1.join()
    p2.join()
