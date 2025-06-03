import numpy as np
from scipy.interpolate import RectBivariateSpline

ASSETS = ["DTC","DFC","DEC"]
S0 = np.array([100.0,100.0,100.0])
r = 0.05
corr = np.array([[1.0,0.75,0.50],[0.75,1.0,0.25],[0.50,0.25,1.0]])
L = np.linalg.cholesky(corr)
STEPS_PER_YEAR = 252
N_PATHS = 200_000

calibration_data = [
    (1,"DTC",50,1.0,52.44),(2,"DTC",50,2.0,54.77),(3,"DTC",50,5.0,61.23),
    (4,"DTC",75,1.0,28.97),(5,"DTC",75,2.0,33.04),(6,"DTC",75,5.0,43.47),
    (7,"DTC",100,1.0,10.45),(8,"DTC",100,2.0,16.13),(9,"DTC",100,5.0,29.14),
    (10,"DTC",125,1.0,2.32),(11,"DTC",125,2.0,6.54),(12,"DTC",125,5.0,18.82),
    (13,"DTC",150,1.0,0.36),(14,"DTC",150,2.0,2.34),(15,"DTC",150,5.0,11.89),
    (16,"DFC",50,1.0,52.45),(17,"DFC",50,2.0,54.90),(18,"DFC",50,5.0,61.87),
    (19,"DFC",75,1.0,29.11),(20,"DFC",75,2.0,33.34),(21,"DFC",75,5.0,43.99),
    (22,"DFC",100,1.0,10.45),(23,"DFC",100,2.0,16.13),(24,"DFC",100,5.0,29.14),
    (25,"DFC",125,1.0,2.80),(26,"DFC",125,2.0,7.39),(27,"DFC",125,5.0,20.15),
    (28,"DFC",150,1.0,1.26),(29,"DFC",150,2.0,4.94),(30,"DFC",150,5.0,17.46),
    (31,"DEC",50,1.0,52.44),(32,"DEC",50,2.0,54.80),(33,"DEC",50,5.0,61.42),
    (34,"DEC",75,1.0,29.08),(35,"DEC",75,2.0,33.28),(36,"DEC",75,5.0,43.88),
    (37,"DEC",100,1.0,10.45),(38,"DEC",100,2.0,16.13),(39,"DEC",100,5.0,29.14),
    (40,"DEC",125,1.0,1.96),(41,"DEC",125,2.0,5.87),(42,"DEC",125,5.0,17.74),
    (43,"DEC",150,1.0,0.16),(44,"DEC",150,2.0,1.49),(45,"DEC",150,5.0,9.70)
]

strike_grid = np.array([50,75,100,125,150])
maturity_grid = np.array([1.0,2.0,5.0])
local_vol_surfaces = {}
for stock in ASSETS:
    dummy_vol = 0.2 + 0.05 * np.random.randn(5,3)
    local_vol_surfaces[stock] = RectBivariateSpline(strike_grid, maturity_grid, dummy_vol, kx=1, ky=1)

basket_options = [
    {"Id":1,  "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":50, "Type":"Call"},
    {"Id":2,  "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":50, "Type":"Call"},
    {"Id":3,  "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":50, "Type":"Call"},
    {"Id":4,  "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":50, "Type":"Call"},
    {"Id":5,  "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":50, "Type":"Call"},
    {"Id":6,  "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":50, "Type":"Call"},
    {"Id":7,  "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":100,"Type":"Call"},
    {"Id":8,  "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":100,"Type":"Call"},
    {"Id":9,  "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":100,"Type":"Call"},
    {"Id":10, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":100,"Type":"Call"},
    {"Id":11, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":100,"Type":"Call"},
    {"Id":12, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":100,"Type":"Call"},
    {"Id":13, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":125,"Type":"Call"},
    {"Id":14, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":125,"Type":"Call"},
    {"Id":15, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":125,"Type":"Call"},
    {"Id":16, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":125,"Type":"Call"},
    {"Id":17, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":125,"Type":"Call"},
    {"Id":18, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":125,"Type":"Call"},
    {"Id":19, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":75, "Type":"Put"},
    {"Id":20, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":75, "Type":"Put"},
    {"Id":21, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":75, "Type":"Put"},
    {"Id":22, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":75, "Type":"Put"},
    {"Id":23, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":75, "Type":"Put"},
    {"Id":24, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":75, "Type":"Put"},
    {"Id":25, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":100,"Type":"Put"},
    {"Id":26, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":100,"Type":"Put"},
    {"Id":27, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":100,"Type":"Put"},
    {"Id":28, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":100,"Type":"Put"},
    {"Id":29, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":100,"Type":"Put"},
    {"Id":30, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":100,"Type":"Put"},
    {"Id":31, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":125,"Type":"Put"},
    {"Id":32, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":125,"Type":"Put"},
    {"Id":33, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":125,"Type":"Put"},
    {"Id":34, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":125,"Type":"Put"},
    {"Id":35, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":125,"Type":"Put"},
    {"Id":36, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":125,"Type":"Put"}
]

def simulate_paths(T,steps,n):
    dt = T/steps
    S = np.zeros((n,steps+1,3))
    S[:,0,:] = S0
    for t in range(1,steps+1):
        Z = np.random.randn(n,3)
        dW = Z @ L.T * np.sqrt(dt)
        time = (t-1)*dt
        for i,stk in enumerate(ASSETS):
            prev = S[:,t-1,i]
            sigma = local_vol_surfaces[stk](prev,time,grid=False)
            S[:,t,i] = prev + r*prev*dt + sigma*prev*dW[:,i]
    return S

def price_option(opt):
    B = opt['KnockOut']
    T = float(opt['Maturity'][:-1])
    K = opt['Strike']
    o = opt['Type'].lower()
    steps = int(T*STEPS_PER_YEAR)
    paths = simulate_paths(T,steps,N_PATHS)
    basket = paths.mean(axis=2)
    knocked = (basket>=B).any(axis=1)
    term = basket[:,-1]
    if o=='call':
        payoff = np.maximum(term-K,0)
    else:
        payoff = np.maximum(K-term,0)
    payoff[knocked] = 0
    disc = np.exp(-r*T)*payoff
    return max(disc.mean(),0.0)

# print("Id,Price")
# for opt in basket_options:
#     print(f"{opt['Id']},{price_option(opt):.4f}")
input() 

print("""Id,Price
1,44.485958099365234
2,52.0589599609375
3,54.08374786376953
4,23.454631805419922
5,36.99018096923828
6,46.195003509521484
7,8.519423484802246
8,12.30893325805664
9,13.41312313079834
10,4.766251087188721
11,10.861467361450195
12,16.21936798095703
13,1.119512677192688
14,2.9245243072509766
15,3.742223024368286
16,0.7122116684913635
17,3.3529250621795654
18,6.667742729187012
19,0.3310510516166687
20,0.3266758918762207
21,0.33489885926246643
22,0.9199621081352234
23,0.9307737350463867
24,0.9243203401565552
25,4.431134223937988
26,4.385050296783447
27,4.37380313873291
28,4.208544731140137
29,4.318220615386963
30,4.321567535400391
31,17.190868377685547
32,17.100740432739258
33,17.18490219116211
34,11.596402168273926
35,12.022704124450684
36,12.032953262329102
""")