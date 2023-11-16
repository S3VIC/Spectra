# no imports

def rectIntegLeft(x, y):
    if len(x) != len(y):
        assert False, "Size of x and y do not match"

    else:
        area = 0
        for i in range(len(x) - 1):
            a = x[i + 1] - x[i]
            b = y[i]
            area = area + a * b

        return area


def rectIntegRight(x, y):
    if len(x) != len(y):
        assert False, "Size of x and y do not match"

    else:
        area = 0
        for i in range(len(x) - 1):
            a = x[i + 1] - x[i]
            b = y[i + 1]
            area = area + a * b

        return area


def trapInteg(x, y):
    if len(x) != len(y):
        assert False, "Size of x and y do not match"

    else:
        area = 0
        for i in range(len(x) - 1):
            h = x[i + 1] - x[i]
            a = y[i + 1]
            b = y[i]
            area = area + h * (a + b)
        return area / 2
