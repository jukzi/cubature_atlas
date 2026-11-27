from math import comb, ceil

def counting_value(d, n, centrally_symmetric=False):
    """Counting value Z of the atlas.

    Plain value: (#monomials of degree <= n) / (unknowns per node = d + 1).
    On a point-symmetric domain (square, cube) a centrally symmetric rule only
    has to meet the E moments of even degree with pairs of nodes, which gives
    ceil(2 E / (d + 1)); Z is the smaller of the two values.
    """
    z = ceil(comb(n + d, d) / (d + 1))
    if centrally_symmetric:
        even = sum(comb(j + d - 1, d - 1) for j in range(0, n + 1, 2))
        z = min(z, ceil(2 * even / (d + 1)))
    return z

def lower_bound(d, n):
    """Degree 2k needs at least dim P_k nodes (any domain)."""
    return comb(n // 2 + d, d)

def lower_bound_square_moeller(n):
    """Moeller bound for odd n = 2k-1 on the square: dim P_{k-1} + floor(k/2)."""
    k = (n + 1) // 2
    return comb(k + 1, 2) + k // 2

def gauss_product(d, n):
    return ceil((n + 1) / 2) ** d

if __name__ == "__main__":
    for d, name, sym in [(1, "interval", True), (2, "triangle", False), (2, "square", True), (3, "tetra", False), (3, "cube", True)]:
        print(name, [(n, counting_value(d, n, sym), lower_bound(d, n), gauss_product(d, n)) for n in (1, 5, 9, 20)])
    print("square, Moeller:", [lower_bound_square_moeller(n) for n in (3, 9, 11, 13, 15, 17, 19)])
