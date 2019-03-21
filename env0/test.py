from multiprocessing.pool import ThreadPool


class Poop:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1

        while self.count < 2:
            pass

        return self.count


pool = ThreadPool(processes=2)

pp = Poop()

x = pool.apply_async(pp.inc)
y = pool.apply_async(pp.inc)

xr = x.get()
yr = y.get()

print(xr, yr)







