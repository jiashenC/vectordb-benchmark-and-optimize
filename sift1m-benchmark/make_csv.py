with open("trainIndex.csv", "w") as f:
    f.write("index\n")
    for i in range(1000000):
        f.write(f"{i}\n")