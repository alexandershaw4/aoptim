function [r,ri] = acorrel(x,y)
% Just a pearson correl coeff
%
% AS

ri = ( (x - mean(x)).*(y-mean(y)) ) ./ ( sqrt( sum( (x-mean(x)).^2 ) * sum( (y-mean(y)).^2 ) ));
r  = sum(ri);