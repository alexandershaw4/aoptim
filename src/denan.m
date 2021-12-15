function y = denan(x,r)
% replace nans in x with r.
% leave out r to use 0
% y = denan(x,r)
% AS

if nargin < 2 || isempty(r)
    r = 0;
end

dx = spm_vec(x);
dx(isnan(dx))= r;
dx(isinf(dx))= r;

y = spm_unvec(dx,x);