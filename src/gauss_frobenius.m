function e = gauss_frobenius(x,y)

% first  pass gauss error
dgY = atcm.fun.QtoGauss(real(x),12*2);
dgy = atcm.fun.QtoGauss(real(y),12*2);
Dg  = dgY - dgy;
e   = trace(Dg'*Dg);