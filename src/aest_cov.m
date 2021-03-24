function C = aest_cov(X,N)

MM = N;%30;                             % delay/embedding
N = length(X);
Y=zeros(N-MM+1,MM);
for m=1:MM
    Y(:,m) = X((1:N-MM+1)+m-1);
end;
Cemb=Y'*Y / (N-MM+1);
C=Cemb;