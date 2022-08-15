function PHI = apolybasis(X,D)
% compute the basis functions for a polynomial regression
% AS

N = length(X);
%D = 4;

PHI=zeros(N,D);
for i=1:N
    for j=1:D
        PHI(i,j)=X(i).^(j-1);
    end
end