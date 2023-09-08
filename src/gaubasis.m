function y = gaubasis(n,k)
% Generate Gaussian basis set of k-components over range n
%
% y = gaubasis(n,k);
% plot(1:n,y)
%
% AS2023

if nargin < 2
    k = n;
end

Q = VtoGauss(ones(n,1));
[u,s,v] = svd(Q);
y = u(:,1:k)'*Q;
y = y./norm(y);
