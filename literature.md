# Literature

Literature on cubature rules with few nodes and positive weights, and on the numerical methods used to find and prove them.
Every entry was looked up in the source (DOI, arXiv or publisher metadata) when it was added; do not add a reference that has not been checked. The notes only say what the abstract, the documentation or the code docstrings say, or what was measured in the atlas project. Slobodkins–Tausch and Taylor–Wingate–Bos were read in full; Diallo–Worku through its arXiv page; all others only by abstract or metadata.

The three sections are:

1. [**Used in the atlas**](#1-used-in-the-atlas): the work is cited in the documentation or the code, or its rules or algorithm are part of the atlas.
2. [**Tried, not worthwhile**](#2-tried-not-worthwhile): the idea was tested and did not pay off. The note says what was tested and what came out.
3. [**Not used yet**](#3-not-used-yet): relevant, but not (or not yet fully) tried.

Each section is sorted by date of appearance, oldest first. An entry has the usual citation with the link to the source, a short note on how it relates to the atlas, and the abstract. Abstracts are copied word for word from arXiv, from the metadata the publisher deposited at Crossref, or from an open-access article. Where only a publisher's page has the abstract and it can be read, the entry gives a short summary in our own words instead; where the page cannot be read at all, the entry says so. The link leads to the source in every case.

## How to extend this file

Every pull request that uses, tests or finds literature adds it here, in the right section and at the right place by date. If an entry moves from section 3 to section 1 or 2, move it. Give the exact outcome in the note (what was tried, what the numbers were). Check the reference against its DOI or arXiv page before adding it. Never invent a citation.

## 1. Used in the atlas

### V. Tchakaloff: Formules de cubature mécanique à coefficients non négatifs. Bull. Sci. Math. (2) 81 (1957) 123–134. <https://arxiv.org/abs/math/0502473>

Tchakaloff's theorem: a positive rule with at most as many nodes as dim P_k exists; the idea behind the non-negative least squares start in `code/simplex_cubature.py`. The paper itself could not be read online; the link is the paper of Bayer and Teichmann (Proc. AMS 134 (2006)), whose bibliography confirms this reference.

> No abstract available

### S. L. Sobolev: The formulas of mechanical cubature on the surface of a sphere. Sibirsk. Mat. Zh. 3 (1962) 769–796. <https://mathnet.ru/eng/smj4866>

Cited in `code/cubature_search.py` for the fact that a rule invariant under a finite group is exact as soon as it is exact for the invariant polynomials.

> No abstract available

### A. H. Stroud: Approximate Calculation of Multiple Integrals. Prentice-Hall, Englewood Cliffs (1971). <https://archive.org/details/approximatecalcu0000stro_b8j7/>

Conical product rule used as the reference quadrature in `code/simplex_cubature.py`. The link is the Internet Archive record of the book (Prentice-Hall, Englewood Cliffs, N.J., 1971, xiii + 431 pages, ISBN 0130438936); the book is access-restricted there, so it can only be borrowed. A review in Math. Comp. 27 (1973) is at <https://doi.org/10.2307/2005635>.

_No abstract: it is a book, and the Internet Archive record has no description._

### H. M. Möller: Kubaturformeln mit minimaler Knotenzahl. Numer. Math. 25 (1976) 185–200. <https://doi.org/10.1007/BF01462272>

Lower bounds for the number of nodes; listed in the references of the documentation (`content/references.json`).

Summary in our own words, the publisher's abstract is not reproduced: for double integrals the paper derives a lower bound for the number of nodes of a cubature formula of degree 2s-1. A formula of that degree attaining the bound exists exactly when a certain condition holds, and under this condition all such minimal formulas can be constructed. Examples and a generalisation to n dimensions are given.

### H. M. Möller: Lower bounds for the number of nodes in cubature formulae. In: G. Hämmerlin (ed.), Numerische Integration, ISNM 45, Birkhäuser, Basel (1979) 221–230. <https://doi.org/10.1007/978-3-0348-6288-2_17>

Lower bound for the number of nodes; listed in the references of the documentation (`content/references.json`).

Summary in our own words, the publisher's abstract is not reproduced: it recalls that Gauss formulae have the fewest nodes among all quadrature formulae of a given degree on an interval, and says that the multidimensional analogue needs lower bounds for the number of nodes together with information on how sharp they are.

### D. A. Dunavant: High degree efficient symmetrical Gaussian quadrature rules for the triangle. Int. J. Numer. Methods Eng. 21 (1985) 1129–1148. <https://doi.org/10.1002/nme.1620210612>

Cited in the documentation as an early symmetric triangle collection.

> Gaussian quadrature is required for the computation of matrices based on the isoparametric formulztion of the finite element method. A brief review of existing quadrature rules for the triangle is given, and the method for the determination of high degree efficient symmetrical rules for the triangle is discussed. New quadrature rules of degree 12–20 are presented, and a short FORTRAN program is included.

### M. Dubiner: Spectral methods on triangles and other domains. J. Sci. Comput. 6 (1991) 345–390. <https://doi.org/10.1007/BF01060030>

Orthonormal Proriol–Koornwinder–Dubiner basis, used for the independent check of the rules in `code/simplex_cubature.py`.

Summary in our own words, the publisher's abstract is not reproduced: the paper obtains multidimensional spectral methods as warped products of one-dimensional ones, which generalises tensor products. Applications are spectral approximation on a disk and on a triangle, with the disk method used for Navier-Stokes simulation in a periodic pipe and the triangle method in a spectral element scheme; the approximation degree of the triangle method is computed in a new way.

### R. Cools, P. Rabinowitz: Monomial cubature rules since “Stroud”: a compilation. J. Comput. Appl. Math. 48 (1993) 309–326. <https://doi.org/10.1016/0377-0427(93)90027-9>

Compilation of known rules; cited in the documentation.

Summary in our own words, the publisher's abstract is not reproduced: a bibliography of references to cubature rules that appeared after Stroud's book (1971), for the standard regions n-cube, n-simplex, n-sphere and the whole space.

### R. Cools: Constructing cubature formulae: the science behind the art. Acta Numerica 6 (1997) 1–54. <https://doi.org/10.1017/S0962492900002701>

Survey of construction methods and of the theory of minimal rules; cited in the documentation.

> In this paper we present a general, theoretical foundation for the construction of cubature formulae to approximate multivariate integrals. The focus is on cubature formulae that are exact for certain vector spaces of polynomials. Our main quality criteria are the algebraic and trigonometric degrees. The constructions using ideal theory and invariant theory are outlined. The known lower bounds for the number of points are surveyed and characterizations of minimal cubature formulae are given. We include references to all known minimal cubature formulae. Finally, some methods to construct cubature formulae illustrate the previously introduced concepts and theorems.

### R. Cools: Monomial cubature rules since “Stroud”: a compilation – part 2. J. Comput. Appl. Math. 112 (1999) 21–27. <https://doi.org/10.1016/S0377-0427(99)00229-0>

Compilation of known rules; cited in the documentation.

> No abstract available

### R. Cools: An encyclopaedia of cubature formulas. J. Complexity 19 (2003) 445–453. <https://doi.org/10.1016/S0885-064X(03)00011-6>

The online database at KU Leuven (https://nines.cs.kuleuven.be/research/ecf/) asks to cite this article together with the two compilations above; cited in the documentation.

Summary in our own words, the publisher's abstract is not reproduced: the author began collecting published cubature formulas for standard regions about 13 years earlier; the paper describes how this collection is made available to a wide audience through the World Wide Web.

### S. Wandzura, H. Xiao: Symmetric quadrature rules on a triangle. Comput. Math. Appl. 45 (2003) 1829–1840. <https://doi.org/10.1016/S0898-1221(03)90004-6>

Symmetric triangle rules; cited in the documentation.

Summary in our own words, the publisher's abstract is not reproduced: the paper constructs quadrature rules on triangles that have positive weights, full symmetry and rapid convergence, comparable to Gauss rules on intervals. A scheme that combines simple group theory with numerical optimisation gives such rules up to order 30; the approach formalises an older one by Lyness and Jespersen and extends to other regions, and example formulae and numerical results are shown.

### A. Hinrichs, E. Novak: Cubature formulas for symmetric measures in higher dimensions with few points. arXiv:math/0509101 (2005). <https://arxiv.org/abs/math/0509101>

Section 2 gives Möller's lower bound for centrally symmetric weights in explicit form; cited in the documentation.

> We study cubature formulas for d-dimensional integrals with an arbitrary symmetric weight function of tensor product form. We present a construction that yields a high polynomial exactness: for fixed degree l=5 or l=7 and large dimension, the number of knots is only slightly larger than the lower bound of Möller and much smaller compared to the known constructions. We also show, for any odd degree l=2k+1, that the minimal number of points is almost independent of the weight function. This is also true for the integration over the (Euclidean) sphere.

### H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663–676. <https://doi.org/10.1016/j.camwa.2009.10.027>

Cited in the docstrings of `code/cubature_search.py` and `code/simplex_cubature.py` for the node elimination idea; the tetrahedron counts in `code/simplex_cubature.py` are taken from its rules as tabulated by J. Burkardt.

Summary in our own words, the publisher's abstract is not reproduced: the paper presents a numerical algorithm that constructs efficient high-order quadrature rules in two and more dimensions. The rules have positive weights and interior nodes, like Gauss rules in one dimension, and can be generated with different degrees of symmetry adapted to the domain. Examples are rules for polynomials on triangles, squares and cubes up to degree 50, which are described as near optimal in the number of nodes, many of them new. (Read on ResearchGate, https://www.researchgate.net/publication/220513533.)

### Z. Meng, Z. Luo: Constructing cubature formulae of degree 5 with few points. J. Comput. Appl. Math. 237 (2013) 354–362 (arXiv:1111.5094, 2011). <https://arxiv.org/abs/1111.5094>

States that for even degrees no lower bound beyond dim P_k is known; cited in the documentation. Abstract of the arXiv version.

> This paper will devote to construct a family of fifth degree cubature formulae for $n$-cube with symmetric measure and $n$-dimensional spherically symmetrical region. The formula for $n$-cube contains at most $n^2+5n+3$ points and for $n$-dimensional spherically symmetrical region contains only $n^2+3n+3$ points. Moreover, the numbers can be reduced to $n^2+3n+1$ and $n^2+n+1$ if $n=7$ respectively, the later of which is minimal.

### F. D. Witherden, P. E. Vincent: On the identification of symmetric quadrature rules for finite element methods. Comput. Math. Appl. 69 (2015) 1232–1241. <https://arxiv.org/abs/1409.1865>

Orbit parametrisation and orthonormal-basis moment equations (`code/cubature_search.py`); the family `witherden_vincent_2015` reproduces its triangle rules. Abstract of the arXiv version.

> In this paper we describe a methodology for the identification of symmetric quadrature rules inside of quadrilaterals, triangles, tetrahedra, prisms, pyramids, and hexahedra. The methodology is free from manual intervention and is capable of identifying an ensemble of rules with a given strength and a given number of points. We also present polyquad which is an implementation of our methodology. Using polyquad we proceed to derive a complete set of symmetric rules on the aforementioned domains. All rules possess purely positive weights and have all points inside the domain. Many of the rules appear to be new, and an improvement over those tabulated in the literature.

### F. Piazzon, A. Sommariva, M. Vianello: Caratheodory-Tchakaloff least squares. 2017 International Conference on Sampling Theory and Applications (SampTA), IEEE (2017) 672–676. <https://doi.org/10.1109/SAMPTA.2017.8024337>

Caratheodory–Tchakaloff subsampling, the idea behind the non-negative least squares start in `code/simplex_cubature.py`. The docstring named a different venue and a DOI that does not resolve; it now cites this paper.

Summary in our own words, the publisher's abstract is not reproduced: the paper discusses Caratheodory-Tchakaloff subsampling, implemented by linear or quadratic programming, to compress multivariate discrete measures and to compress polynomial least squares problems.

### J. Jaśkowiec, N. Sukumar: High-order cubature rules for tetrahedra. Int. J. Numer. Methods Eng. 121 (2020) 2418–2436. <https://doi.org/10.1002/nme.6313>

Fully symmetric tetrahedron rules, family `jaskowiec_sukumar_2020`.

> In this paper, we construct new high‐order numerical integration schemes for tetrahedra, with positive weights and integration points that are in the interior of the domain. The construction of cubature rules is a challenging problem, which requires the solution of strongly nonlinear algebraic (moment) equations with side conditions given by affine inequality constraints. We present a robust algorithm based on a sequence of three modified Newton procedures to solve the constrained minimization problem. In the literature, numerical integration rules for the tetrahedron are available up to order
                    p
                    =15. We obtain integration rules for the tetrahedron from
                    p
                    =2 to
                    p
                    =20, which are computed using multiprecision arithmetic. For
                    p
                    ≤15, our approach provides integration rules that have the same or fewer number of integration points than existing rules; for
                    p
                    =16 to
                    p
                    =20, our rules are new. Numerical tests are presented that verify the polynomial‐precision of the cubature rules. Convergence studies are performed for the integration of exponential, rational, weakly singular and trigonometric test functions over tetrahedra with flat and curved faces. In all tests, improvements in accuracy is realized as
                    p
                    is increased, though in some cases nonmonotonic convergence is observed.

### P. D. Brubeck, Y. Nakatsukasa, L. N. Trefethen: Vandermonde with Arnoldi. SIAM Review 63 (2021) 405–415. <https://doi.org/10.1137/19M130100X>

Idea of the Arnoldi-type construction of the polynomial basis in `code/simplex_cubature.py`.

> Vandermonde matrices are exponentially ill-conditioned, rendering the familiar “polyval(polyfit)” algorithm for polynomial interpolation and least-squares fitting ineffective at higher degrees. We show that Arnoldi orthogonalization fixes the problem. This amounts to on-the-fly construction of discrete orthogonal polynomials by Stieltjes orthogonalization.

### J. Jaśkowiec, N. Sukumar: High-order symmetric cubature rules for tetrahedra and pyramids. Int. J. Numer. Methods Eng. 122 (2021) 148–171. <https://doi.org/10.1002/nme.6528>

Fully symmetric tetrahedron rules up to degree 20, family `jaskowiec_sukumar_2020`.

> In this article, we present an algorithm to construct high‐order fully symmetric cubature rules for tetrahedral and pyramidal elements, with positive weights and integration points that are in the interior of the domain. Cubature rules are fully symmetric if they are invariant to affine transformations of the domain. We divide the integration points into symmetry orbits where each orbit contains all the points generated by the permutation stars. These relations are represented by equality constraints. The construction of symmetric cubature rules require the solution of nonlinear polynomial equations with both inequality and equality constraints. For higher orders, we use an algorithm that consists of five sequential phases to produce the cubature rules. In the literature, symmetric numerical integration rules are available for the tetrahedron for orders p = 1–10, 14, and for the pyramid up to p = 10. We have obtained fully symmetric cubature rules for both of these elements up to order p = 20. Numerical tests are presented that verify the polynomial‐precision of the cubature rules. Convergence studies are performed for the integration of exponential, weakly singular, and trigonometric test functions over both elements with flat and curved faces. With increase in p, improvements in accuracy is realized, though nonmonotonic convergence is observed.

### J. Jaśkowiec, N. Sukumar: Addendum to the paper “High-order symmetric cubature rules for tetrahedra and pyramids”. Int. J. Numer. Methods Eng. 122 (2021) 1875–1883. <https://doi.org/10.1002/nme.6600>

Addendum to the previous entry.

> In Jaśkowiec and Sukumar (Int J Numer Methods Eng, doi: 10.1002/nme.6528, 2020), we presented high‐order (p = 2–20) symmetric cubatures rules for tetrahedra and pyramids. This algorithm was sensitive to the initial location of the cubature nodes, and it did not converge for p > 11 over prisms and hexahedra (cubes). In this addendum, we resolve this issue and obtain high‐order symmetric rules over prisms and cubes. For the prism, we use the initial guess for the cubature rule as the tensor product of a cubature rule over a triangle and a univariate Gauss quadrature rule, and for the cube the initial guess is the tensor product of univariate Gauss quadrature rules. Verification and convergence tests are presented to affirm the accuracy of the cubature rules. On applying the cubature algorithm described in Jaśkowiec and Sukumar (Int J Numer Methods Eng, 121(11), 2418–2436, 2020), we also construct nonsymmetric high‐order (p = 2–20) cubature rules over prisms, cubes, and pyramids. In the supplementary materials, all cubature rules (128 digits of precision) are provided in a text file and in  format.

### G. Chuluunbaatar, O. Chuluunbaatar, A. A. Gusev, S. I. Vinitsky: PI-type fully symmetric quadrature rules on the 3-, …, 6-simplexes. Comput. Math. Appl. 124 (2022) 89–97. <https://doi.org/10.1016/j.camwa.2022.08.016>

Fully symmetric tetrahedron counts up to degree 20, tabulated for comparison in `code/simplex_cubature.py` (as quoted in Table 2 of Worku–Hicken–Zingg, arXiv:2409.02027: 79, 98, 123, 145, 175, 209, 248, 284, 343, 383, 441 at degrees 10 to 20). They enter the column N_Lit of the tetrahedron. The rules of the triangle and the tetrahedron up to degree 20 are stored in the family `chuluunbaatar_2022`, converted from the compact orbit form of the supplementary material of the paper (full text and supplement read) and proven with interval arithmetic. How the rules were found (full text read): fully symmetric systems (for the tetrahedron 72, 94 and 108 equations at degrees 17, 19 and 20), orbit combinations chosen so that the number of unknowns exceeds the number of equations by at most d−1 and filtered by consistency conditions, box-constrained Levenberg–Marquardt from random starts, in quadruple precision and refined in Maple to 50 digits. The orbit structure of the tetrahedron rules at degrees 17, 19 and 20 is 8, 2, 14, 3 orbits S31, S22, S211, S1111 (284 nodes) and 1, 7, 3, 18, 5 orbits S4, S31, S22, S211, S1111 (383 nodes) and 1, 8, 4, 20, 6 (441 nodes).

Summary in our own words, the publisher's abstract is not reproduced: the paper gives fully symmetric quadrature rules with positive weights and nodes inside the simplex (PI type): up to order 20 on the tetrahedron, 16 on the 4-simplex and 10 on the 5- and 6-simplexes. The node counts do not exceed the known ones and most rules are new. They were computed with modified Levenberg-Marquardt methods for nonlinear equations with convex constraints, first in FORTRAN to 25 digits and then refined to 50 digits in MAPLE.

### A. Slobodkins, J. Tausch: A node elimination algorithm for cubature of high-dimensional polytopes. Comput. Math. Appl. 144 (2023) 229–236 (arXiv:2207.10737, 2022). <https://arxiv.org/abs/2207.10737>

Ranking of the nodes by the norm of the minimum-norm linearised correction, used in `code/` and in the GPU search. Its tetrahedron counts for degrees 15, 17 and 23 are below the earlier atlas counts and served as targets. Abstract of the arXiv version.

> Node elimination is a numerical approach to obtain cubature rules for the approximation of multivariate integrals. Beginning with a known cubature rule, nodes are selected for elimination, and a new, more efficient rule is constructed by iteratively solving the moment equations. This paper introduces a new criterion for selecting which nodes to eliminate that is based on a linearization of the moment equation. In addition, a penalized iterative solver is introduced, that ensures that weights are positive and nodes are inside the integration domain. A strategy for constructing an initial quadrature rule for various polytopes in several space dimensions is described. High efficiency rules are presented for two, three and four dimensional polytopes. The new rules are compared with rules that are obtained by combining tensor products of one dimensional quadrature rules and domain transformations, as well as with known analytically constructed cubature rules.

### Y. Xu: Minimal cubature rules and Koornwinder polynomials. arXiv:2407.09903 (2024). <https://arxiv.org/abs/2407.09903>

Theorem 3.1 states Möller's bound with the structural matrices of the three-term relation; cited in the documentation.

> In his classical paper [5], Koornwinder studied a family of orthogonal polynomials of two variables, derived from symmetric polynomials. This family possesses a rare property that orthogonal polynomials of degree $n$ have $n(n+1)/2$ real common zeros, which leads to important examples in the theory of minimal cubature rules. This paper aims to give an account of the minimal cubature rules of two variables and examples originating from Koornwinder polynomials, and we will also provide further examples.

### Z. A. Worku, J. E. Hicken, D. W. Zingg: Very high-order symmetric positive-interior quadrature rules on triangles and tetrahedra. J. Comput. Appl. Math. 472 (2026) 116782 (arXiv:2409.02027, 2024). <https://arxiv.org/abs/2409.02027>

Fully symmetric rules up to degree 84 (triangle) and 40 (tetrahedron); comparison counts and elimination idea cited in `code/simplex_cubature.py`. Abstract of the arXiv version.

> We present novel fully-symmetric quadrature rules with positive weights and strictly interior nodes of degrees up to 84 on triangles and 40 on tetrahedra. Initial guesses for solving the nonlinear systems of equations needed to derive quadrature rules are generated by forming tensor-product structures on quadrilateral/hexahedral subdomains of the simplices using the Legendre-Gauss nodes on the first half of the line reference element. In combination with a methodology for node elimination, these initial guesses lead to the development of highly efficient quadrature rules, even for very high polynomial degrees. Using existing estimates of the minimum number of quadrature points for a given degree, we show that the derived quadrature rules on triangles and tetrahedra are more than 95% and 80% efficient, respectively, for almost all degrees. The accuracy of the quadrature rules is demonstrated through numerical examples.

### M. Diallo, Z. A. Worku: High-order symmetric positive interior quadrature rules on two and three dimensional domains. arXiv:2601.14488 (2026). <https://arxiv.org/abs/2601.14488>

Fully symmetric square and cube rules, family `diallo_worku_2026`; priority order of the orbit elimination in `code/cubature_search.py`.

> Fully symmetric positive interior (f-SPI) quadrature rules are key building blocks for high-order discretizations of partial differential equations, yet high-degree rules with few nodes remain scarce on reference elements commonly used in mesh generation. We construct new f-SPI rules on the square, cube, prism, and pyramid by coupling a variable parameterization that enforces positivity and interiority with an efficient Levenberg-Marquardt optimization and a symmetry-aware node-reduction strategy that eliminates and collapses orbits, allowing transitions between symmetry types. The resulting rules achieve degrees up to 77 on the square, 45 on the cube, and 30 on the prism and pyramid, and for most degrees use fewer nodes than previously published f-SPI quadrature rules. Verification tests demonstrate comparable accuracy to existing rules. Complete node and weight data are also provided.

### J. Pinkse: Positive weight Hermite and Legendre quadrature rules. arXiv:2609.26840 (2026). <https://arxiv.org/abs/2609.26840>

80-digit positive-weight Gauss–Hermite and interior-node Gauss–Legendre rules up to five dimensions (data: doi:10.5281/zenodo.22881864, CC BY 4.0). Its counts enter the literature column N_Lit of the atlas (squares up to degree 77, cubes up to degree 45; the smaller of its count and the older literature counts). Rules that the atlas could prove with interval arithmetic are reproduced in the family `pinkse_2026`: the cube at degrees 21, 25, 29, 31, 33, 35, 37, 39, 41 and 43, and the square at degree 35. For the other square degrees (31, 33, 37 to 77) and for the cube at degree 45 the interval proof did not succeed (ill-conditioned or too large for the present proof program); their counts stand in N_Lit without a stored rule. Its Table 2 records, for every degree, the best earlier count and where it was found (read from the full text); N_Lit takes these as cited: for the square the earlier counts up to degree 25 (Stroud 1971, Möller 1976, Cools–Haegemans 1988, Festa–Sommariva 2012), 132 at degree 27 and 152 at degree 29 (Festa–Sommariva, Xiao–Gimbutas), for the cube 6, 13 (Stroud 1971) and 26, 50, 84, 130, 190 at degrees 7 to 15 (Xiao–Gimbutas 2010); from degree 17 the earlier cube counts are those of Diallo–Worku (2026). Pinkse is credited in N_Lit only where his count is smaller than all of these (or equal and not older). Abstract of the arXiv version.

> This paper contains new 80-digit positive-weight Gauss-Hermite and positive-weight interior-node Gauss-Legendre quadrature rules for up to five dimensions and varying polynomial degree accuracy (depending on quadrature type and dimension). Some of these rules improve on the best available rules in the literature and some offer rules where none (other than the tensor product) existed. The results were produced by combining methodology developed by previous researchers with new approaches. The full precision rules themselves are at https://doi.org/10.5281/zenodo.22881864. Software in Julia, Python, and R providing the rules is available via a package registry and/or GitHub: Quadriceps.jl (both 64-bit and 128-bit; https://github.com/NittanyLion/Quadriceps.jl), quadriceps-py (64-bit only; https://github.com/NittanyLion/quadriceps-py), and quadriceps-r (64-bit only; https://github.com/NittanyLion/quadriceps-r). The software used to create these rules is available via PositiveWeightQuadratureSolvers.jl (https://github.com/NittanyLion/PositiveWeightQuadratureSolvers.jl). The replication package is at QuadricepsReplicationPackage.jl (https://github.com/NittanyLion/QuadricepsReplicationPackage.jl). A snapshot of all five packages is archived at https://doi.org/10.5281/zenodo.22883240.

### Online collections (no date)

A. Sommariva: Cubature rules on the triangle (simplex). University of Padova, online sets with 26 cited sources; the page shows no date. <https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/rules_triangle.html> Online collection used for comparison.

J. Burkardt: Quadrature rule collections (cube_arbq_rule, tetrahedron_jaskowiec_rule, triangle_wandzura_rule and more). Online collection. <https://people.sc.fsu.edu/~jburkardt/m_src/cube_arbq_rule/cube_arbq_rule.html> Online collection used for comparison.

## 2. Tried, not worthwhile

### G. H. Golub, V. Pereyra: The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. SIAM J. Numer. Anal. 10 (1973) 413–432. <https://doi.org/10.1137/0710036>

Variable projection: the weights are eliminated by a linear least squares solve. Benchmarked on 2026-11-12 against plain Levenberg–Marquardt on random starts (same batch and iteration cap, GPU and CPU); clearly worse in converged rules per second. Not used.

> For given data $(t_i ,y_i ),i = 1, \cdots ,m$, we consider the least squares fit of nonlinear models of the form \[ \eta ({\bf a},{\boldsymbol \alpha} ;t) = \sum _{j = 1}^n {a_j \varphi _j ({\boldsymbol \alpha} ;t),\qquad {\bf a} \in \mathcal{R}^n ,\qquad {\boldsymbol \alpha} \in \mathcal{R}^k .} \] For this purpose we study the minimization of the nonlinear functional \[ r({\bf a},{\boldsymbol \alpha} ) = \sum\limits_{i = 1}^m {\left( {y_i - \eta \left( {{\bf a},{\boldsymbol \alpha} ,t_i } \right)} \right)^2 } . \] It is shown that by defining the matrix $\{ {\bf \Phi} ({\boldsymbol \alpha} )\} _{i,j} = \varphi _j ({\boldsymbol \alpha} ;t_i )$, and the modified functional $r_2 ({\boldsymbol \alpha} ) = \| {\bf y} - {\bf \Phi} ({\boldsymbol \alpha} ){\bf \Phi} ^ + ({\boldsymbol \alpha} ){\bf y} \|_2^2 $, it is possible to optimize first with respect to the parameters ${\boldsymbol \alpha} $, and then to obtain, a posteriors, the optimal parameters $\bf {\hat a}$. The matrix ${\bf \Phi} ^ + ({\boldsymbol{\alpha}} )$ is the Moore–Penrose generalized inverse of ${\bf \Phi} ({\boldsymbol{\alpha}} )$. We develop formulas for the Frechet derivative of orthogonal projectors associated with ${\bf \Phi} ({\boldsymbol{\alpha}} )$ and also for ${\bf \Phi} ^ + ({\boldsymbol{\alpha}} )$, under the hypothesis that ${\bf \Phi} ({\boldsymbol{\alpha}} )$ is of constant (though not necessarily full) rank. Detailed algorithms are presented which make extensive use of well-known reliable linear least squares techniques, and numerical results and comparisons are given. These results are generalizations of those of H. D. Scolnik [20] and Guttman, Pereyra and Scolnik [9].

### L. Kaufman: A variable projection method for solving separable nonlinear least squares problems. BIT 15 (1975) 49–57. <https://doi.org/10.1007/BF01932995>

The cheaper Jacobian of variable projection (Kaufman's approximation) was part of the same benchmark; same outcome. Not used.

Summary in our own words, the publisher's abstract is not reproduced: the paper treats separable nonlinear least squares problems, in which a linear and a nonlinear set of parameters appear. Golub and Pereyra reduced the problem to one nonlinear problem in the nonlinear parameters only. The paper proposes a new way of determining these parameters that proved computationally more efficient than the Golub-Pereyra scheme.

### M. A. Taylor, B. A. Wingate, L. P. Bos: Several new quadrature formulas for polynomial integration in the triangle. arXiv:math/0501496 (2005). <https://arxiv.org/abs/math/0501496>

Asymmetric triangle rules. Their node counts were compared with the atlas on 2026-11-12 and none is below the atlas count (for example degree 13: 36 nodes against 35 in the atlas; degree 25: 120 against 117). Nothing to add. Abstract of the arXiv version.

> We present several new quadrature formulas in the triangle for exact integration of polynomials. The points were computed numerically with a cardinal function algorithm which imposes that the number of quadrature points $N$ be equal to the dimension of a lower dimensional polynomial space. Quadrature forumulas are presented for up to degree $d=25$, all which have positive weights and contain no points outside the triangle. Seven of these quadrature formulas improve on previously known results.

## 3. Not used yet

### G. Alefeld, A. Gienger, F. Potra: Efficient numerical validation of solutions of nonlinear systems. SIAM J. Numer. Anal. 31 (1994) 252–260. <https://doi.org/10.1137/0731013>

Numerical validation of solutions of nonlinear systems. `tools/verify_rigorous.py` already contains a Krawczyk test; this paper has not been compared with it.

> A new stopping criterion for Newton’s method is derived by combining the properties of the Krawczyk operator and a corollary of the Newton–Kantorovich theorem. When this criterion is satisfied the authors use the last three Newton iterates to compute an interval vector that is very likely to contain a solution of the given nonlinear system. The existence of such a solution is tested using Krawczyk’s operator. Furthermore, each element from this interval vector considered as an approximation to the solution has a relative error that is of the order of the machine precision. Extensive numerical testing has shown that the proposed method has very good practical performance.

### J. Eriksson, P.-Å. Wedin, M. E. Gulliksson, I. Söderkvist: Regularization methods for uniformly rank-deficient nonlinear least-squares problems. J. Optim. Theory Appl. 127 (2005) 1–26. <https://doi.org/10.1007/s10957-005-6389-0>

Found in the literature search for rank-deficient nonlinear least squares problems; only title and metadata were seen, so it is not known yet whether it helps.

Summary in our own words, the publisher's abstract is not reproduced: Levenberg-Marquardt and similar methods run into trouble when the Jacobian is rank-deficient or very ill-conditioned at the solution. The paper studies uniformly rank-deficient problems, where the Jacobian has the same deficient rank near a solution, so the solution is not locally unique, and presents two techniques, a minimum-norm solution and Tikhonov regularisation, with optimality conditions, algorithms, convergence results and numerical experiments. The ideas extend to more general rank-deficient or ill-conditioned problems.

### M. A. Taylor, B. A. Wingate, L. P. Bos: A cardinal function algorithm for computing multivariate quadrature points. SIAM J. Numer. Anal. 45 (2007) 193–205. <https://doi.org/10.1137/050625801>

Different construction of quadrature points (cardinal functions instead of node elimination), according to its title and abstract.

> We present a new algorithm for numerically computing quadrature formulas for arbitrary domains which exactly integrate a given polynomial space. An effective method for constructing quadrature formulas has been to numerically solve a nonlinear set of equations for the quadrature points and their associated weights. Symmetry conditions are often used to reduce the number of equations and unknowns. Our algorithm instead relies on the construction of cardinal functions and thus requires that the number of quadrature points N be equal to the dimension of a prescribed lower dimensional polynomial space. The cardinal functions allow us to treat the quadrature weights as dependent variables and remove them, as well as an equivalent number of equations, from the numerical optimization procedure. We give results for the triangle, where for all degrees $d \le 25$, we find quadrature formulas of this form which have positive weights and contain no points outside the triangle. Seven of these quadrature formulas improve on previously known results.

### M. A. Taylor: Asymmetric cubature formulas for polynomial integration in the triangle and square. J. Comput. Appl. Math. 218 (2008) 184–191. <https://doi.org/10.1016/j.cam.2007.04.050>

Asymmetric rules on the triangle and square, like the atlas rules (full text read). Five rules with positive weights and no points outside, each with fewer points than the best earlier ones in Cools's database: triangle degrees 10, 11, 12 with 24, 27, 32 points; square degrees 10 and 12 with 22 and 31 points. These counts enter the column N_Lit (the same triangle counts appear in Table 1 of Xiao–Gimbutas 2010, which is later).

Summary in our own words, the publisher's abstract is not reproduced: the paper presents five new cubature formulas for exact integration of polynomials: degrees 10, 11 and 12 on the triangle and degrees 10 and 12 on the square. The points were computed with a cardinal function algorithm that imposes no symmetry. The formulas have positive weights, no points outside the domain, and fewer points than previously known results.

### J. W. Helton, J. Nie: A semidefinite approach for truncated K-moment problems. arXiv:1105.0410 (2011). <https://arxiv.org/abs/1105.0410>

Flat-extension test for the moment problem. Theoretically interesting, but the semidefinite programs look far too large for rules with several hundred nodes (judged from the abstract, not tested). Abstract of the arXiv version.

> A truncated moment sequence (tms) of degree d is a vector indexed by monomials whose degree is at most d. Let K be a semialgebraic set.The truncated K-moment problem (TKMP) is: when does a tms y admit a positive Borel measure supported? This paper proposes a semidefinite programming (SDP) approach for solving TKMP. When K is compact, we get the following results: whether a tms y of degree d admits a K-measure or notcan be checked via solving a sequence of SDP problems; when y admits no K-measure, a certificate will be given; when y admits a K-measure, a representing measure for y would be obtained from solving the SDP under some necessary and some sufficient conditions. Moreover, we also propose a practical SDP method for finding flat extensions, which in our numerical experiments always finds a finitely atomic representing measure for a tms when it admits one.

### I. C. F. Ipsen, C. T. Kelley, S. R. Pope: Rank-deficient nonlinear least squares problems and subset selection. SIAM J. Numer. Anal. 49 (2011) 1244–1266. <https://doi.org/10.1137/090780882>

Treats stagnation of Levenberg–Marquardt on ill-conditioned problems and proposes subset selection; a possible addition to the LM step.

> We examine the local convergence of the Levenberg–Marquardt method for the solution of nonlinear least squares problems that are rank-deficient and have nonzero residual. We show that replacing the Jacobian by a truncated singular value decomposition can be numerically unstable. We recommend instead the use of subset selection. We corroborate our recommendations by perturbation analyses and numerical experiments.

### M. K. Transtrum, J. P. Sethna: Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization. arXiv:1201.5885 (2012). <https://arxiv.org/abs/1201.5885>

Geodesic acceleration and Broyden updates could save Jacobian evaluations, which are the bottleneck at large moment counts. Abstract of the arXiv version.

> When minimizing a nonlinear least-squares function, the Levenberg-Marquardt algorithm can suffer from a slow convergence, particularly when it must navigate a narrow canyon en route to a best fit. On the other hand, when the least-squares function is very flat, the algorithm may easily become lost in parameter space. We introduce several improvements to the Levenberg-Marquardt algorithm in order to improve both its convergence speed and robustness to initial parameter guesses. We update the usual step to include a geodesic acceleration correction term, explore a systematic way of accepting uphill steps that may increase the residual sum of squares due to Umrigar and Nightingale, and employ the Broyden method to update the Jacobian matrix. We test these changes by comparing their performance on a number of test problems with standard implementations of the algorithm. We suggest that these two particular challenges, slow convergence and robustness to initial guesses, are complimentary problems. Schemes that improve convergence speed often make the algorithm less robust to the initial guess, and vice versa. We provide an open source implementation of our improvements that allow the user to adjust the algorithm parameters to suit particular needs.

### M. Festa, A. Sommariva: Computing almost minimal formulas on the square. J. Comput. Appl. Math. 236 (2012) 4296–4302. <https://doi.org/10.1016/j.cam.2012.05.021>

Node elimination on the square, cited by later work next to Xiao–Gimbutas. Its counts for degrees 15, 17, 19 and 23 (43, 54, 67, 96 points) (Table 1); its counts for degrees 24 to 30 are 109, 113, 127, 132, 146, 153, 167 points (Table 2, full text read). Table 1 also lists counts of earlier rules with their types (full text of Omelyan–Solovyan 2006, Tables 1 and 7, read as well): positive interior rules of Stroud (1971, degrees 3, 5, 7: 4, 7, 12), Möller (1976, degree 9: 17), Cools–Haegemans (1988, degrees 11, 13: 24, 33) and Omelyan–Solovyan (2006, degrees 15, 17, 19, 21: 44, 56, 68, 81). Rules with negative weights or nodes outside (44, 57, 68, 85 at degrees 15 to 21; 100 and 112 at degree 23) are not counted. The column N_Lit of the square uses the positive interior counts, credited to the earliest publication; the square column of Xiao–Gimbutas (2010) is now taken from their Table 3 (non-symmetric rules, for example 45 at degree 15, 57 at 17, 71 at 19), and the cube column from their Table 4 (13 at degree 5, 54 at degree 9, 206 at degree 15; the rule of degree 14 has a negative weight).

Summary in our own words, the publisher's abstract is not reproduced: the paper introduces a numerical method for computing cubature rules on the square [-1,1]^2 that are almost minimal, that is with few points for the degree of precision. The rules for all degrees from 24 to 55 are new, and new rules are also given for degrees 15, 17, 19 and 23, with 43, 54, 67 and 96 points.

### B. Vioreanu, V. Rokhlin: Spectra of multiplication operators as a numerical tool. SIAM J. Sci. Comput. 36 (2014) A267–A288. <https://doi.org/10.1137/110860082>

A different route to quadratures, according to its title and abstract.

> We introduce a numerical procedure for the construction of interpolation and quadrature formulae on bounded convex regions in the plane. The construction is based on the behavior of spectra of certain multiplication operators and leads to nodes which are inside a prescribed convex region in ${\mathbb R}^2$. The resulting interpolation schemes are numerically stable and the quadrature formulae have positive weights and almost (but not quite) optimal numbers of nodes. The performance of the algorithm is illustrated by several numerical examples.

### S.-A. Papanicolopulos: Efficient computation of cubature rules with application to new asymmetric rules on the triangle. J. Comput. Appl. Math. 304 (2016) 73–83. <https://doi.org/10.1016/j.cam.2016.03.013>

Asymmetric rules on the triangle. Only title and metadata were seen; the counts have not been compared with the atlas.

Summary in our own words, the publisher's abstract is not reproduced: the paper presents a method for computing cubature rules based on least-squares minimisation and orthogonal bases, applicable to any integration domain. It is tested on asymmetric rules on the triangle, including how to optimise the computation of the basis and its derivatives, and gives three new rules with fewer points than known rules of the same degree.

### A. Abdelfattah, A. Haidar, S. Tomov, J. Dongarra: Performance tuning and optimization techniques of fixed and variable size batched Cholesky factorization on GPUs. Procedia Computer Science 80 (2016) 119–130. <https://doi.org/10.1016/j.procs.2016.05.303>

Batched Cholesky factorisation on GPUs for many small systems; possibly relevant to the minimum-norm step of the search, which factorises one matrix per candidate.

> Solving a large number of relatively small linear systems has recently drawn more attention in the HPC community, due to the importance of such computational workloads in many scientific applications, including sparse multifrontal solvers. Modern hardware accelerators and their architecture require a set of optimization techniques that are very different from the ones used in solving one relatively large matrix. In order to impose concurrency on such throughput-oriented architectures, a common practice is to batch the solution of these matrices as one task offloaded to the underlying hardware, rather than solving them individually.This paper presents a high performance batched Cholesky factorization on large sets of relatively small matrices using Graphics Processing Units (GPUs), and addresses both fixed and variable size batched problems. We investigate various algorithm designs and optimization techniques, and show that it is essential to combine kernel design with performance tuning in order to achieve the best possible performance. We compare our approaches against state-of-the-art CPU solutions as well as GPU-based solutions using existing libraries, and show that, on a K40c GPU for example, our kernels are more than 2 faster.

### P. Breiding, K. Rose, S. Timme: Certifying zeros of polynomial systems using interval arithmetic. arXiv:2011.05000 (2020). <https://arxiv.org/abs/2011.05000>

Krawczyk's method for certifying solutions of square polynomial systems in HomotopyContinuation.jl. Our proof in `tools/verify_rigorous.py` uses a Krawczyk test for an underdetermined system; this paper has not been compared with it. Only the abstract was read. Abstract of the arXiv version.

> We establish interval arithmetic as a practical tool for certification in numerical algebraic geometry. Our software HomotopyContinuation.jl now has a built-in function certify, which proves the correctness of an isolated nonsingular solution to a square system of polynomial equations. The implementation rests on Krawczyk's method. We demonstrate that it dramatically outperforms earlier approaches to certification. We see this contribution as powerful new tool in numerical algebraic geometry, that can make certification the default and not just an option.

### J. Glaubitz: Construction and application of provable positive and exact cubature formulas. IMA J. Numer. Anal. 43 (2023) 1616–1652 (arXiv:2108.02848, 2021). <https://arxiv.org/abs/2108.02848>

Least squares construction of cubature formulas that are provably positive and exact, with only basic linear algebra. Related to the positivity part of our interval proof, but it constructs rules rather than certifying given ones; only the abstract was read. Abstract of the arXiv version.

> Many applications require multi-dimensional numerical integration, often in the form of a cubature formula. These cubature formulas are desired to be positive and exact for certain finite-dimensional function spaces (and weight functions). Although there are several efficient procedures to construct positive and exact cubature formulas for many standard cases, it remains a challenge to do so in a more general setting. Here, we show how the method of least squares can be used to derive provable positive and exact formulas in a general multi-dimensional setting. Thereby, the procedure only makes use of basic linear algebra operations, such as solving a least squares problem. In particular, it is proved that the resulting least squares cubature formulas are ensured to be positive and exact if a sufficiently large number of equidistributed data points is used. We also discuss the application of provable positive and exact least squares cubature formulas to construct nested stable high-order rules and positive interpolatory formulas. Finally, our findings shed new light on some existing methods for multivariate numerical integration and under which restrictions these are ensured to be successful.

### Z. A. Worku, J. E. Hicken, D. W. Zingg: Quadrature rules on triangles and tetrahedra for multidimensional summation-by-parts operators. J. Sci. Comput. 101 (2024), arXiv:2311.15576 (2023). <https://arxiv.org/abs/2311.15576>

Fully symmetric positive-weight rules on triangles and tetrahedra with matching facet nodes for diagonal-E operators; the counts are above the PI counts of the atlas and the rules are symmetric. Only the abstract was read. Abstract of the arXiv version.

> Multidimensional diagonal-norm summation-by-parts (SBP) operators with collocated volume and facet nodes, known as diagonal-$ \mathsf{E} $ operators, are attractive for entropy-stable discretizations from an efficiency standpoint. However, there is a limited number of such operators, and those currently in existence often have a relatively high node count for a given polynomial order due to a scarcity of suitable quadrature rules. We present several new symmetric positive-weight quadrature rules on triangles and tetrahedra that are suitable for construction of diagonal-$ \mathsf{E} $ SBP operators. For triangles, quadrature rules of degree one through twenty with facet nodes that correspond to the Legendre-Gauss-Lobatto (LGL) and Legendre-Gauss (LG) quadrature rules are derived. For tetrahedra, quadrature rules of degree one through ten are presented along with the corresponding facet quadrature rules. All of the quadrature rules are provided in a supplementary data repository. The quadrature rules are used to construct novel SBP diagonal-$ \mathsf{E} $ operators, whose accuracy and maximum timestep restrictions are studied numerically.

### A. Schwarz, A. Anders, C. Brower, H. Bayraktar, J. Gunnels, K. Clark, R. G. Xu, S. Rodriguez, S. Cayrols, P. Tabaszewski, V. Podlozhnyuk: Guaranteed DGEMM accuracy while using reduced precision tensor cores through extensions of the Ozaki scheme. arXiv:2511.13778 (2025); SCA/HPCAsia 2026. <https://arxiv.org/abs/2511.13778>

FP64 emulation on tensor cores. It might speed up the slow float64 polish, but according to its abstract it targets large matrix products, and it has not been tested on consumer GPUs. Abstract of the arXiv version.

> The rapid growth of artificial intelligence (AI) has made low-precision formats such as FP16, FP8, and, most recently, block-scaled FP4 the primary focus of modern GPUs, where Tensor Cores now deliver orders-of-magnitude higher throughput than traditional FP64 pipelines. This hardware shift has sparked a new line of algorithm research: using low-precision units to emulate double-precision accuracy through schemes such as Ozaki decompositions. We advance this direction with Automatic Dynamic Precision (ADP), a fully GPU-resident framework that makes emulated FP64 matrix multiplication both efficient and reliable. At its core is the Exponent Span Capacity (ESC), a hardware-agnostic estimator that conservatively determines the decomposition parameter (also known as slices) required to achieve FP64-level accuracy. Built on ESC, ADP integrates exception handling, run time heuristics, and seamless fallback to native FP64, ensuring correctness without host-device synchronization or user intervention. Additionally, we further improve Ozaki-style decompositions with an unsigned integer slicing scheme, which increases representational efficiency and reduces computational waste. Validated against recently proposed BLAS grading tests, ADP consistently preserves FP64 fidelity on challenging inputs while incurring less than 10% run time overhead. In a 55-bit mantissa setting, our approach achieves up to 2.3x and 13.2x speedups over native FP64 GEMM on NVIDIA Blackwell GB200 and the RTX Pro 6000 Blackwell Server Edition, respectively. Our results demonstrate that low-precision accelerators can serve as a practical, production-ready foundation for high-fidelity and high-performance scientific computing workloads.

### M. Schäfer, T. Ullrich: Beyond Tchakaloff quadrature: positive functionals, frames and widths. arXiv:2511.15425 (2025). <https://arxiv.org/abs/2511.15425>

Generalises Tchakaloff's theorem (existence of exact rules with non-negative weights) and gives bounds for the Tchakaloff quadrature widths, that is the smallest possible node counts. For tetrahedron degrees the guaranteed number is dim P_k (2925 for degree 24), far above the rules found here, so the bounds say little for us; only the abstract was read. Abstract of the arXiv version.

> Tchakaloff's theorem from 1957 asserts the existence of exact quadrature rules with non-negative weights for any polynomial space of finite degree on $\mathbb{R}^d$ if the underlying measure is positive, compactly supported, and absolutely continuous with respect to the Lebesgue measure. This classical result coined the term Tchakaloff quadrature for quadrature that is exact and only uses non-negative weights. It has been a long-standing endeavor, under which conditions such rules exist. A final answer was given in 2012 by Bisgaard with the insight that, in fact, every finite-dimensional space of integrable functions on a positive measure space admits them. In this article we recall this result and provide a major extension to the question of positive discretizability of $\mathbb{C}$-linear functionals on finite-dimensional spaces. We introduce the notion of strict $S$-positivity for such functionals, where $S$ are subsets of the functional's domain, and show the equivalence of positive discretizability to being strictly $S$-positive for a suitable choice of $S$. We further investigate consequences for other discretization problems. One fundamental implication is the guaranteed existence of $L_p$-Marcinkiewicz-Zygmund equalities in finite-dimensional spaces of $p$-integrable functions in case that $p$ is an even integer, another the exact discretizability of any frame in $\mathbb{K}^n$, where $\mathbb{K}\in\{\mathbb{R},\mathbb{C}\}$, if a rescaling of the frame elements is allowed. In addition, we provide bounds for Tchakaloff quadrature widths $κ_n^+$ and, addressing the question of constructibility of discretization points, establish a connection to $D$-optimal design.
