"""import_burkardt_rules: literature rules whose nodes and weights are printed in public rule files, brought into the atlas format (one rule file each).
Source files (data tables of published rules, one Fortran subroutine per degree, 16 digits):
  square_os: Omelyan, Solovyan (2006), Tables 2-5, rotationally symmetric square rules of the degrees 15, 17, 19, 21 (generators with 32 digits, embedded in this program).
  triangle_xg: Xiao, Gimbutas (2010), fully symmetric triangle rules of the degrees 0 to 50 (file triangle_symq_rule.f90, barycentric nodes a, b, c and weights w).
The numbers are read from the printed arrays; nothing is recomputed or tuned. The nodes (b, c) are the Cartesian coordinates on the atlas triangle (0,0), (1,0), (0,1); the weights are scaled to the sum 1/2.
The program stops with an error if the node count differs from the count in reference_data/literature_counts.json, if the moment error is larger than 1e-12, if a weight is not positive or if a node lies outside the domain.
Usage: python code/import_burkardt_rules.py --source triangle_xg --file <triangle_symq_rule.f90> --degrees 32 33 35 [--root .]
Afterwards: python tools/verify_rigorous.py --strict --save <rule files>; python tools/build_atlas.py --base origin/main --changed
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cubature_atlas as C
import po_elim

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[DdEe][-+]?\d+)?")


def read_subroutines(path):
    """Return {subroutine name: {array name: [floats]}} for all arrays given as name = (/ ... /) or data name / ... /."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    kept = []
    for line in lines:
        s = line.strip()
        if s.startswith("!"):
            continue
        kept.append(s.replace("&", " "))
    text = "\n".join(kept)
    chunks = re.split(r"(?im)^\s*subroutine\s+", text)
    result = {}
    for chunk in chunks[1:]:
        name = re.match(r"\s*(\w+)", chunk).group(1).lower()
        arrays = {}
        for m in re.finditer(r"(?is)(\w+)\s*=\s*\(/(.*?)/\)", chunk):
            arrays[m.group(1).lower()] = parse_numbers(m.group(2))
        for m in re.finditer(r"(?is)\bdata\s+(\w+)\s*/(.*?)/", chunk):
            arrays[m.group(1).lower()] = parse_numbers(m.group(2))
        result[name] = arrays
    return result


def parse_numbers(text):
    return [float(t.replace("D", "e").replace("d", "e")) for t in NUMBER.findall(text)]


def literature_count(root, domain, key, degree):
    path = os.path.join(root, "reference_data", "literature_counts.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)["domains"][domain]
    for entry in data:
        if entry["k"] == key:
            return entry["arr"][degree - 1]
    return None


def triangle_xg(subs, degree):
    """Fully symmetric triangle rule of the given degree: barycentric a, b, c and weights w."""
    arrays = subs["rule%02d" % degree]
    a = np.array(arrays["a"])
    b = np.array(arrays["b"])
    c = np.array(arrays["c"])
    w = np.array(arrays["w"])
    X = np.stack([b, c], axis=1)
    inside = float(min(a.min(), b.min(), c.min()))
    scale = float(w.sum()) / 0.5
    return X, w / scale, inside, scale


def read_data_post(path, degree):
    """Arrays xs<degree>, ys<degree>, ws<degree> of the block '%%% nmax = <degree>' in a file data_post.m (35 digits, all nodes listed)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    blocks = re.split(r"(?m)^%%% nmax = ", text)
    for block in blocks[1:]:
        if int(re.match(r"\d+", block).group(0)) != degree:
            continue
        out = {}
        for name in ("xs", "ys", "ws"):
            m = re.search(r"%s%d = \[(.*?)\];" % (name, degree), block, re.S)
            out[name] = np.array([float(t) for t in m.group(1).replace(",", " ").split()])
        return out
    raise SystemExit("degree %d not found in %s" % (degree, path))


def triangle_xg_rot(path, degree):
    """Rotationally symmetric Xiao-Gimbutas triangle rule (file tables/triarotq/data_post.m of the authors' repository) on the triangle
    (-1,-1/sqrt3), (1,-1/sqrt3), (0,2/sqrt3); mapped affinely to (0,0), (1,0), (0,1), weights scaled to the sum 1/2."""
    d = read_data_post(os.path.join(path, "tables", "triarotq", "data_post.m"), degree)
    P = np.stack([d["xs"], d["ys"]], axis=1)
    w = d["ws"]
    s3 = math.sqrt(3.0)
    v0 = np.array([-1.0, -1.0 / s3])
    v1 = np.array([1.0, -1.0 / s3])
    v2 = np.array([0.0, 2.0 / s3])
    m = np.array([v1 - v0, v2 - v0]).T
    X = np.linalg.solve(m, (P - v0).T).T
    inside = float(min(X.min(), (1.0 - X.sum(axis=1)).min()))
    scale = float(w.sum()) / 0.5
    return X, w / scale, inside, scale


def tetra_xg(subs, degree):
    """Xiao-Gimbutas tetrahedron rule on the tetrahedron with the vertices
    (-1,-1/sqrt3,-1/sqrt6), (0,2/sqrt3,-1/sqrt6), (1,-1/sqrt3,-1/sqrt6), (0,0,3/sqrt6); mapped affinely to (0,0,0), (1,0,0), (0,1,0), (0,0,1)."""
    arrays = subs["rule%02d" % degree]
    P = np.stack([arrays["xs"], arrays["ys"], arrays["zs"]], axis=1)
    w = np.array(arrays["ws"])
    s3 = math.sqrt(3.0)
    s6 = math.sqrt(6.0)
    v0 = np.array([-1.0, -1.0 / s3, -1.0 / s6])
    v1 = np.array([0.0, 2.0 / s3, -1.0 / s6])
    v2 = np.array([1.0, -1.0 / s3, -1.0 / s6])
    v3 = np.array([0.0, 0.0, 3.0 / s6])
    m = np.array([v1 - v0, v2 - v0, v3 - v0]).T
    X = np.linalg.solve(m, (P - v0).T).T
    inside = float(min(X.min(), (1.0 - X.sum(axis=1)).min()))
    scale = float(w.sum()) / (1.0 / 6.0)
    return X, w / scale, inside, scale


def box_rule(names, volume):
    """Reader of a rule on [-1,1]^d given as arrays named by `names` and a weight array `ws`."""
    def reader(subs, degree):
        key = "rule%02d" % degree
        arrays = subs[key]
        X = np.stack([arrays[n] for n in names], axis=1)
        w = np.array(arrays["ws"])
        inside = float(1.0 - np.abs(X).max())
        scale = float(w.sum()) / volume
        return X, w / scale, inside, scale
    return reader


def smr_square(subs, degree):
    """Festa-Sommariva rule smrNN: a flat array of triples (x, y, w) on [-1,1]^2."""
    arrays = subs["smr%02d" % degree]
    flat = np.array(arrays["xyw"])
    t = flat.reshape(-1, 3)
    X = t[:, :2]
    w = t[:, 2]
    inside = float(1.0 - np.abs(X).max())
    scale = float(w.sum()) / 4.0
    return X, w / scale, inside, scale


# Omelyan, Solovyan (2006), Tables 2 to 5: generators (w, x, y) of the rotationally symmetric rules of the degrees 15, 17, 19, 21 (32 digits as printed).
# The nodes of a generator are (x, y), (-y, x), (-x, -y), (y, -x), each with the weight w; the generator (0, 0) is one node whose weight is the printed value
# (the paper writes w0 = 4 w_M for the four collapsed nodes; the printed number in Table 5 already is the weight of the single node: with 4 times the printed value the moment error is 3e-2, with the printed value 1e-16).
OS_TABLES = {
    15: """
0.20881470204497523521771058289754E-1 0.98798456650771809034922121236542 0.77126821223875533899886933446485
0.25545901574497276542640153395248E-1 0.90815949600657000212015099547736 0.95703183434690690872237176442598
0.31203866624933300871149690867662E-1 0.67928365833453304991325391003325 0.88260197593087253601344445274335
0.38010761595074827467645518285610E-1 0.50911373411758353514778637359508 0.97120312974183699854692313856226
0.41449061852426148002787373214871E-1 0.97675332466910190352385200798077 0.83559862608781647288448365846813E-1
0.79320407004083334044710201039891E-1 0.75619936719149244012005066365912 0.75619936719149244012005066365912
0.88901265758751523303980720079987E-1 0.89778569328633877480008574782677 0.46676265923796434848237353795400
0.12016982158206027507823569713056 0.20599307074252141729418963873618 0.84079448454078540426562160968883
0.16882043410639799754153511014621 0.45144312511299139017533875886564 0.56245686233219940637540066298377
0.16987162497336185160489786440053 0.66683824538360873834071399129626 0.19046630243571720761679616635243
0.21582538472391594202064661314968 0.74295704755765822553432311307323E-1 0.32397702249753019818251854432752
""",
    17: """
0.10693483986974526468925667171638E-1 0.83395914422050762595707520328329 0.99690134998258294114169765276688
0.16771622989325482379964908559201E-1 0.96701240760377864958940778776159 0.92101565015369642619085740576658
0.21520834803173017585623196363995E-1 0.98651441086033068583570701960810 0.56251780244667252352153081833715
0.24893201532665059892584476318209E-1 0.17035060808995408160378979789427E-1 0.98204914256843033449712481990261
0.42463258472030940473501779894230E-1 0.42523896522453030941022314252505 0.95796453236865195741899809390262
0.53711265037645010830029221647514E-1 0.88194109089215356624316729768736 0.74117688569732509081061208741447
0.54579479693382460849318673206721E-1 0.69978907719058600912388980227382 0.88728024293255774907349992409998
0.67375653622461385504403959336007E-1 0.90858412958838344797723547617253 0.25843122151820770173022364732347
0.98025282885102299426881768287454E-1 0.73933759205292015806620475085721 0.47295583257297618882242532468520
0.98325651584666601742691443856667E-1 0.21246155837885419289305793553062 0
0.10576898319665727200249259468733 0.15644172095846342514322716437912 0.81259212523912311994061744159953
0.10593453574575401283483020818374 0.50398563819427997044830482422504 0.68201093297792530795370853308359
0.14990405484916921950315250394890 0.58243895074467257078462336257058 0.11846544560647891209927499369579
0.15003269160099271050559959853839 0.28970323065541272212557470108155 0.40876985953794338411643329876836
""",
    19: """
0.42157189312457273371400997162954E-2 0.93790944060174636724164305692148 0.99998546072852907260205380652647
0.99237601014741223600089798896376E-2 0.59787966519157168521483636741590 0.98732847941781540087026300838966
0.15078678879581549295330034135223E-1 0.96865781747798834472115427804246 0.89837495163532572964949889543009
0.15121496864822956266676863866018E-1 0.98871713276447330663578670732177 0.62259634389530287767136776393609
0.23821621047339582724750103846036E-1 0.98398534132338386681313796251589 0.81109463487824943619725131193050E-1
0.28746437252189671047629194361029E-1 0.36585775934555586089438861852973 0.96462400422891970516096923553526
0.31348715503861464721826917341708E-1 0.78392149176096602345110934186655 0.93756444544378175318295617860873
0.49762852666717116332578404045926E-1 0.86937144898957875204792277623522 0.74380866034597101765279618918913
0.55534775159604041101829554407116E-1 0.91889377777573801032232199904313 0.38410875822737883260638757649426
0.65013710432173970207381031565964E-1 0.14078396804456848628363687508206 0.87987957307981017658529064786666
0.73819068900731731823980126564345E-1 0.56379666815446526573028097449771 0.81879210607636015929368719928237
0.90385825968150641564727715959435E-1 0.76200274293070327698230061865243 0.15749920122732269808023356410127
0.91881833336425013923433485117405E-1 0.69745388342191267884442122044750 0.54657527460177776847334916572487
0.94967142638856099564605740671378E-1 0.52310392339404494392757182766006 0
0.10498174724102843457467466433826 0.30566836903929191370033838487568 0.65017670270687960549278395428873
0.11871336850928058424347422704556 0.44898642628288082765338494532756 0.34240465380680230945939322837053
0.12668324656651729290995285712867 0.98253418759835132805054567507033E-1 0.23509621629115532325303789785547
""",
    21: """
0.59245289910274777823163684444026E-2 0.99742844318071465788852153329446 0.52349333540342268677302187698119
0.63181879993530976749052313712955E-2 0.90092205722857715090631770295789 0.99213624611198765984022577926158
0.77654356771885822575525989789117E-2 0.98137581661152322617866583081353 0.93577766500519228442627576803182
0.13313236524649387539261801099149E-1 0.47562398846061921636952360049609 0.98806503981406364167006543205075
0.18028023632000303112294832251383E-1 0.68396217875524370909034162562144E-1 0.98289142121346795894575007552095
0.20759211084811453614957408545828E-1 0.96445171170290974019893928634261 0.73079564192792033198229729840349
0.24345594686962939048122829629801E-1 0.96789534067760540982739074113386 0.28416943704022564251542547384397
0.27291829813157459363923440908365E-1 0.71204626183902144323409345553631 0.95001737577395264422880843244573
0.31313724489320677669626239379042E-1 0.86686503563504933746419509606443 0.85754514904426211818642071770359
0.43028684463625497581941353350492E-1 0.29388895765490255861532312924984 0.91478375374420147743639913929870
0.47435107497760585342967184111271E-1 0.88245289720509646533706783064117 0.78310571319347967606817150072422E-1
0.49535232791099913534195616930148E-1 0.88356465549777630296872390851348 0.50650027819745540446699219481203
0.59086609737522776429493659734483E-1 0.53714330328796591598235477256257 0.82111662321535962574448188332070
0.61598898755573131953283218720393E-1 0.73297737278688849035966507531475 0.67811474157990352881988795983159
0.68662882737105569078247834904061E-1 0.74340815897994389367005251095709 0.25891212405917942502949658613377
0.77511025760449779132628565445376E-1 0.14352379867257862891820869134808 0.73714284444878233387764988974016
0.77655900861398148704502327066780E-1 0.57716374806487034338151001622839 0.42605137832252378501649531823938
0.84165159143253389201574044182295E-1 0.38010653105519745774291045326124 0.59190980196005469388929534014808
0.11664395742356559711226466539621 0.49380304750704567296572150088764 0.46968438389915362845722235027390E-1
0.12587077966701428595400393561138 0.23300694276964919884562248755586 0.27854054992870057594151995625227
0.13498395305263979164774737575571 0 0
""",
}


def square_os(subs, degree):
    """Omelyan-Solovyan rule from OS_TABLES: rotationally symmetric, positive weights, nodes inside [-1,1]^2, weights summing to 4."""
    rows = [[float(t) for t in line.split()] for line in OS_TABLES[degree].strip().splitlines()]
    nodes = []
    weights = []
    for w, x, y in rows:
        if x == 0.0 and y == 0.0:
            nodes.append((0.0, 0.0))
            weights.append(w)
            continue
        for p in ((x, y), (-y, x), (-x, -y), (y, -x)):
            nodes.append(p)
            weights.append(w)
    X = np.array(nodes)
    w = np.array(weights)
    inside = float(1.0 - np.abs(X).max())
    scale = float(w.sum()) / 4.0
    return X, w / scale, inside, scale


def tetra_wk(path, degree):
    """Worku-Hicken-Zingg tetrahedron rule: file tet_q<degree>_n<N>_ext.dat in the directory `path`; three header lines, then one block per node
    (x, y, z, w on separate lines, blank line between blocks) on the tetrahedron with the vertex coordinates -1 and 1; mapped by x -> (x+1)/2."""
    names = [n for n in os.listdir(path) if n.startswith("tet_q%d_n" % degree) and n.endswith("_ext.dat")]
    if len(names) != 1:
        raise SystemExit("degree %d: expected one file, found %s" % (degree, names))
    with open(os.path.join(path, names[0]), encoding="utf-8") as fh:
        lines = [t.strip() for t in fh.read().splitlines()]
    numbers = [float(t) for t in lines[3:] if t]
    t = np.array(numbers).reshape(-1, 4)
    X = (t[:, :3] + 1.0) / 2.0
    w = t[:, 3] / 8.0
    inside = float(min(X.min(), (1.0 - X.sum(axis=1)).min()))
    scale = float(w.sum()) / (1.0 / 6.0)
    return X, w / scale, inside, scale


def cube_pk(path, degree):
    """Pinkse rule: file legendre_d3_p<degree>_q<q>_n<N>.csv in the directory `path` (header lines starting with #, then a line x1,x2,x3,w and one line per node)
    on the unit cube with the weights summing to 1; mapped by x -> 2x - 1, weights times 8."""
    names = [n for n in os.listdir(path) if n.startswith("legendre_d3_p%d_q" % degree) and n.endswith(".csv")]
    if len(names) != 1:
        raise SystemExit("degree %d: expected one file, found %s" % (degree, names))
    rows = []
    with open(os.path.join(path, names[0]), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("x1") or not line.strip():
                continue
            rows.append([float(t) for t in line.split(",")])
    t = np.array(rows)
    X = 2.0 * t[:, :3] - 1.0
    w = 8.0 * t[:, 3]
    inside = float(1.0 - np.abs(X).max())
    scale = float(w.sum()) / 8.0
    return X, w / scale, inside, scale



def xg_table(table, domain):
    """Reader for tables/<table>/data_post.m of the authors' repository (35 digits, all nodes listed, block 'nmax = <degree>').
    triangle: (-1,-1/sqrt3), (1,-1/sqrt3), (0,2/sqrt3) -> (0,0), (1,0), (0,1), weights to 1/2; tetra: reference tetrahedron of the authors -> (0,0,0), (1,0,0), (0,1,0), (0,0,1), weights to 1/6;
    square and cube: [-1,1]^d unchanged, weights to 4 and 8."""
    def reader(path, degree):
        d = read_data_post_any(os.path.join(path, "tables", table, "data_post.m"), degree)
        names = ["xs", "ys", "zs"][:{"triangle": 2, "square": 2, "tetra": 3, "cube": 3}[domain]]
        P = np.stack([d[n] for n in names], axis=1)
        w = d["ws"]
        s3 = math.sqrt(3.0)
        s6 = math.sqrt(6.0)
        if domain == "triangle":
            v0 = np.array([-1.0, -1.0 / s3])
            vs = [np.array([1.0, -1.0 / s3]), np.array([0.0, 2.0 / s3])]
            volume = 0.5
        elif domain == "tetra":
            v0 = np.array([-1.0, -1.0 / s3, -1.0 / s6])
            vs = [np.array([0.0, 2.0 / s3, -1.0 / s6]), np.array([1.0, -1.0 / s3, -1.0 / s6]), np.array([0.0, 0.0, 3.0 / s6])]
            volume = 1.0 / 6.0
        else:
            X = P
            inside = float(1.0 - np.abs(X).max())
            volume = 4.0 if domain == "square" else 8.0
            scale = float(w.sum()) / volume
            return X, w / scale, inside, scale
        m = np.array([v - v0 for v in vs]).T
        X = np.linalg.solve(m, (P - v0).T).T
        inside = float(min(X.min(), (1.0 - X.sum(axis=1)).min()))
        scale = float(w.sum()) / volume
        return X, w / scale, inside, scale
    return reader


def read_data_post_any(path, degree):
    """Arrays xs, ys, (zs), ws of the block '%%% nmax = <degree>' in a file data_post.m."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for block in re.split(r"(?m)^%%% nmax = ", text)[1:]:
        if int(re.match(r"\d+", block).group(0)) != degree:
            continue
        out = {}
        for name in ("xs", "ys", "zs", "ws"):
            m = re.search(r"%s%d = \[(.*?)\];" % (name, degree), block, re.S)
            if m:
                out[name] = np.array([float(t) for t in m.group(1).replace(",", " ").split()])
        return out
    raise SystemExit("degree %d not found in %s" % (degree, path))


XG_TABLES = [
    ("triarotq", "triangle", "T2", "rotation (group of order 3)"),
    ("triaarbq", "triangle", "T2", "none"),
    ("squarearbq", "square", "C2", "auto"),
    ("squarerotq", "square", "C2", "auto"),
    ("squarerefq", "square", "C2", "auto"),
    ("tetraarbq", "tetra", "T3", "none"),
    ("tetrarotq", "tetra", "T3", "none"),
    ("tetrasym4q", "tetra", "T3", "none"),
    ("cubearbq", "cube", "C3", "auto"),
    ("cuberefq", "cube", "C3", "auto"),
]
XG_SOURCE = ("H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; "
             "rule of the authors' published data tables (more recent than the paper, fewer nodes than the count of the paper), mapped affinely to the atlas domain, weights scaled to its volume.")


def central_symmetry(X, W):
    """True if the node set and the weights are invariant under x -> -x (up to 1e-13)."""
    for p, w in zip(X, W):
        d = np.abs(X + p).max(axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-13 or abs(W[j] - w) > 1e-13 * max(1.0, abs(w)):
            return False
    return True


SOURCES = {
    "tetra_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "tetra",
        "letter": "T3",
        "key": "xg",
        "reader": tetra_xg,
        "symmetry": "none",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; numbers as printed (16 digits) in J. Burkardt's tetrahedron_arbq_rule, mapped affinely from the tetrahedron with the vertices (-1,-1/sqrt3,-1/sqrt6), (0,2/sqrt3,-1/sqrt6), (1,-1/sqrt3,-1/sqrt6), (0,0,3/sqrt6) to the atlas tetrahedron, weights scaled to the sum 1/6.",
    },
    "cube_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "cube",
        "letter": "C3",
        "key": "xg",
        "reader": box_rule(["xs", "ys", "zs"], 8.0),
        "symmetry": "none",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; numbers as printed (16 digits) in J. Burkardt's cube_arbq_rule on [-1,1]^3, weights scaled to the sum 8.",
    },
    "square_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "square",
        "letter": "C2",
        "key": "xg",
        "reader": box_rule(["xs", "ys"], 4.0),
        "symmetry": "none",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; numbers as printed (16 digits) in J. Burkardt's square_arbq_rule on [-1,1]^2, weights scaled to the sum 4.",
    },
    "cube_pk": {
        "family": "pinkse_2026",
        "tag": "PK",
        "domain": "cube",
        "letter": "C3",
        "key": "pk",
        "reader": cube_pk,
        "symmetry": "auto",
        "author": "Pinkse",
        "source": "J. Pinkse: Positive weight Hermite and Legendre quadrature rules. arXiv:2609.26840 (2026); data: Zenodo, doi:10.5281/zenodo.22881864 (CC BY 4.0), transformed from the unit cube to [-1,1]^3 (weights times 8), double precision as published.",
    },
    "tetra_wk": {
        "family": "worku_hicken_zingg_2024",
        "tag": "WHZ",
        "domain": "tetra",
        "letter": "T3",
        "key": "wk",
        "reader": tetra_wk,
        "symmetry": "full",
        "author": "Worku–Hicken–Zingg",
        "source": "Z. A. Worku, J. E. Hicken, D. W. Zingg: Very high-order symmetric positive-interior quadrature rules on triangles and tetrahedra. J. Comput. Appl. Math. 472 (2026) 116782, arXiv:2409.02027; data: github.com/OptimalDesignLab/SummationByParts.jl, pi_quadrature_data/high_precision (MIT license), mapped from the simplex with vertex coordinates -1 and 1 to the unit simplex (x -> (x+1)/2, weights divided by 2^d) and rounded to double precision.",
    },
    "square_os": {
        "family": "omelyan_solovyan_2006",
        "tag": "OS",
        "domain": "square",
        "letter": "C2",
        "key": "os",
        "reader": square_os,
        "symmetry": "rotation (group of order 4)",
        "author": "Omelyan-Solovyan",
        "source": "I. P. Omelyan, V. B. Solovyan: Improved cubature formulae of high degrees of exactness for the square. J. Comput. Appl. Math. 188 (2006) 190-204, doi:10.1016/j.cam.2005.04.005 (Tables 2-5: generators with 32 digits as printed, expanded by the rotation group of order 4; in Table 5 the printed weight of the generator (0,0) is the weight of the central node).",
    },
    "square_fs": {
        "family": "festa_sommariva_2012",
        "tag": "FS",
        "domain": "square",
        "letter": "C2",
        "key": "fs",
        "reader": smr_square,
        "symmetry": "auto",
        "author": "Festa-Sommariva",
        "source": "M. Festa, A. Sommariva: Computing almost minimal formulas on the square. J. Comput. Appl. Math. 236 (2012) 4296-4302; numbers as printed (16 digits) in J. Burkardt's square_minimal_rule (GNU LGPL), which reproduces the authors' sets.",
    },
    "triangle_xg_rot": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "triangle",
        "letter": "T2",
        "key": "xg",
        "reader": triangle_xg_rot,
        "symmetry": "rotation (group of order 3)",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; rotationally symmetric rule triarotq as tabulated (35 digits) by the authors, mapped affinely from the triangle (-1,-1/sqrt3), (1,-1/sqrt3), (0,2/sqrt3) to the atlas triangle, weights scaled to the sum 1/2.",
    },
    "triangle_xg": {
        "family": "xiao_gimbutas_2010",
        "tag": "XG",
        "domain": "triangle",
        "letter": "T2",
        "key": "xg",
        "reader": triangle_xg,
        "symmetry": "full",
        "author": "Xiao-Gimbutas",
        "source": "H. Xiao, Z. Gimbutas: A numerical algorithm for the construction of efficient quadrature rules in two and higher dimensions. Comput. Math. Appl. 59 (2010) 663-676; fully symmetric rule as printed (16 digits) in J. Burkardt's triangle_symq_rule, barycentric nodes (b, c) used as Cartesian coordinates of the atlas triangle, weights scaled to the sum 1/2.",
    },
}



for _t, _dom, _letter, _sym in XG_TABLES:
    SOURCES["xgt_" + _t] = {"family": "xiao_gimbutas_2010", "tag": "XG", "domain": _dom, "letter": _letter, "key": "xg", "reader": xg_table(_t, _dom), "symmetry": _sym,
                            "author": "Xiao-Gimbutas", "source": XG_SOURCE, "better": True}

# Data file and retrieval date, written into the field "source" of every rule ({degree} is replaced).
RETRIEVED = "2026-11-20"
BURKARDT = "https://people.sc.fsu.edu/~jburkardt/f_src/%s/%s.f90"
DATA = {
    "triangle_xg": "J. Burkardt, " + BURKARDT % ("triangle_symq_rule", "triangle_symq_rule") + ", subroutine rule{degree}",
    "tetra_xg": "J. Burkardt, " + BURKARDT % ("tetrahedron_arbq_rule", "tetrahedron_arbq_rule") + ", subroutine rule{degree}",
    "cube_xg": "J. Burkardt, " + BURKARDT % ("cube_arbq_rule", "cube_arbq_rule") + ", subroutine rule{degree}",
    "square_xg": "J. Burkardt, " + BURKARDT % ("square_arbq_rule", "square_arbq_rule") + ", subroutine rule{degree}",
    "square_fs": "J. Burkardt, " + BURKARDT % ("square_minimal_rule", "square_minimal_rule") + ", subroutine smr{degree}",
    "triangle_xg_rot": "github.com/zgimbutas/triasymq (version 1.11, commit 332f5f4), file tables/triarotq/data_post.m, block nmax = {degree}",
    "tetra_wk": "github.com/OptimalDesignLab/SummationByParts.jl (branch master), file pi_quadrature_data/high_precision/tet/expanded/tet_q{degree}_n<N>_ext.dat",
    "cube_pk": "Zenodo doi:10.5281/zenodo.22881864, archive le.zip, file le/rules/legendre_d3_p{degree}_q<q>_n<N>.csv",
    "square_os": "the tables 2 to 5 of the paper (doi:10.1016/j.cam.2005.04.005, page of the publisher, open archive)",
}
for _t, _dom, _letter, _sym in XG_TABLES:
    DATA["xgt_" + _t] = "github.com/zgimbutas/triasymq (version 1.11, commit 332f5f4), file tables/" + _t + "/data_post.m, block nmax = {degree}"


def source_text(cfg, name, degree):
    """Citation, data file and retrieval date of a rule."""
    return "%s Data file: %s. Retrieved %s." % (cfg["source"], DATA[name].replace("{degree}", "%02d" % degree if "subroutine" in DATA[name] else str(degree)), RETRIEVED)


def build_rule(cfg, degree, root, subs, max_p, name):
    X, W, inside, scale = cfg["reader"](subs, degree)
    n = len(W)
    expected = literature_count(root, cfg["domain"], cfg["key"], degree)
    if cfg.get("better"):
        if expected is None or n >= expected:
            raise SystemExit("degree %d: %d nodes, not fewer than the literature count %s" % (degree, n, expected))
    elif expected is not None and expected != n:
        raise SystemExit("degree %d: %d nodes, literature count %s" % (degree, n, expected))
    if W.min() <= 0:
        raise SystemExit("degree %d: weight not positive" % degree)
    if inside < -1e-15:
        raise SystemExit("degree %d: node outside the domain (%g)" % (degree, inside))
    err = po_elim.monomial_error(cfg["domain"], degree, X, W)
    print("degree %d: %d nodes, weight sum factor %.3e, moment error %.2e, min barycentric %.2e" % (degree, n, scale, err, inside), flush=True)
    if err > 1e-12:
        raise SystemExit("degree %d: moment error %.2e too large" % (degree, err))
    symmetry = cfg["symmetry"]
    if symmetry == "auto":
        symmetry = "central" if central_symmetry(X, W) else "none"
    unknowns = n * (X.shape[1] + 1)
    nc = None
    if unknowns <= max_p:
        cert = C.certificate(cfg["domain"], degree, X, W)
        nc = {"N": int(n), "M": int(cert["M"]), "p": int(cert["p"]), "res": float(cert["resid"]), "sigma": float(cert["sigma"]),
              "h": float(cert["h"]), "rad": float(cert["radius"]), "cert": bool(cert["certified"]), "inside": bool(cert["inside"])}
    rid = "CA-%s-%02d-%d-PI-%s" % (cfg["letter"], degree, n, cfg["tag"])
    rule = {"id": rid, "family": cfg["family"], "domain": cfg["domain"], "dimension": X.shape[1], "degree": degree, "points": int(n),
            "quality": "PI", "origin": "literature",
            "method": "taken from the literature (printed numbers read from the data table, weights scaled to the domain volume), moment equations re-checked by the build",
            "author": cfg["author"], "source": source_text(cfg, name, degree), "status": "proven", "symmetry": symmetry}
    if nc is not None:
        rule["numerical_check"] = nc
    rule["nodes"] = [[float(repr(float(v))) for v in row] for row in X]
    rule["weights"] = [float(v) for v in W]
    return rule


def write_rule(root, r):
    d = os.path.join(root, "rules", r["family"], r["domain"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, r["id"] + ".json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--file", default="", help="data file or directory (not needed for sources with the table in this program)")
    ap.add_argument("--degrees", type=int, nargs="+", required=True)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    ap.add_argument("--max-p", type=int, default=4000, help="no numerical_check for rules with more unknowns than this (the check is slow)")
    args = ap.parse_args()
    cfg = SOURCES[args.source]
    subs = None if not args.file else (args.file if os.path.isdir(args.file) else read_subroutines(args.file))
    for degree in args.degrees:
        path = write_rule(args.root, build_rule(cfg, degree, args.root, subs, args.max_p, args.source))
        print("  wrote", path, flush=True)


if __name__ == "__main__":
    main()
