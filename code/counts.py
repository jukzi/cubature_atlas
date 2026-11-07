from math import comb, ceil

def counting_value(d, n):
    """Heuristic: (#monomials of degree <= n) / (unknowns per node = d + 1)."""
    return ceil(comb(n + d, d) / (d + 1))

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
    for d, name in [(1, "interval"), (2, "triangle/square"), (3, "tetra/cube")]:
        print(name, [(n, counting_value(d, n), lower_bound(d, n), gauss_product(d, n)) for n in (1, 5, 9, 20)])
    print("square, Moeller:", [lower_bound_square_moeller(n) for n in (3, 9, 11, 13, 15, 17, 19)])
