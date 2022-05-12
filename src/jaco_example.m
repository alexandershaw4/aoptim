
% generate an exponential function x*exp(1/8) and some data x
f = @(x) x.*exp(1./x);
x = -10:100;

% compute numerical derivatives if f
j = jaco(f,x(:),ones(size(x(:)))/8,0,2);

% note for this function x-diag(j) = f( x - diag(j) ),  
% thus finding f'(x) = f(x)
figure; plot( x-diag(j), f(x(:) - diag(j)) )