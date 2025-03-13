import numpy as np

size = 100000
a = np.random.randint(2, size=size)
b = np.random.randint(2, size=size)

with open("xor.txt", "w") as f: 
    f.write(f"{size}\n")
    for i in range(size): 
        f.write(f"{a[i]} {b[i]} {a[i]^b[i]}\n")

