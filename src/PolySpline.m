function yq = PolySpline(x, y, xq, N)    
% N-dim polynomial regression/spline
%
% AS

X = apolybasis(x,N);

w = x(:)./sum(x(:));

coff = X\(y(:)./w(:));

yq = apolybasis(xq,N)*(coff);


end