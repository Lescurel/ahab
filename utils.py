
if __name__ == "__main__":
    count=0
    c = 0
    with open('mobydick.txt') as og, open("prediction.txt") as pred:
        original = og.read()
        prediction = pred.read()
        for o, p in zip(original, prediction):
            print("{} : {}".format(o,p))
            if o != p:
               count += 1
            c += 1
    print(count)
    print("mean : {}".format(count/c))