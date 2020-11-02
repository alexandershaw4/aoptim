function xh = dirtyml(dx,x,v)
% optimise the parameter estimates dx by optimising their probability 
% according to initial conditions (priors; a and b) and error -
%
% arg max p(P|a,b & e)
%
%

x   = real(x);
dx  = real(dx);
y   = prob(dx,x,v);
fun = @(x0) prob(x0,x,v);
h   = 1/8;

% partial derivatives
for i = 1:length(dx)
    ddx      = dx;
    ddx(i)   = dx(i) + ( dx(i) * v(i) );
    gradx(i) = ( fun(ddx) - fun(dx) );
end

% update
gradx = gradx(:);
xh = dx+(v.*gradx);


end

function [y,pt] = prob(dx,x,v)
% computes 1-pdf as a surrogate for p(x ∈ X) where X is defined by the
% initial conditions (prior mu and std)
for i = 1:length(x)
    if v(i)
        vv     = real(( v(i) ));
        if vv <= 0 || isnan(vv) || isinf(vv); vv = 1/64; end
        pd(i)  = makedist('normal','mu', x(i),'sigma', vv);
        pt(i)  = 1 - pdf( pd(i), dx(i) );
    end
end

y = sum(pt).^2;

end