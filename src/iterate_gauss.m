function [parts,mom] = iterate_gauss(y,k)

y = y(:); y0 = y;
y = abs(y);

for i = 1:k

    p = atcm.fun.indicesofpeaks(y);
    
    G = VtoGauss(y);
    
    x = G(p,:);
    
    b = atcm.fun.lsqnonneg(x',y);
    
    X = b.*x;

    % update residual 
    xp = sum(X,1);
    y  = y - xp(:);

    XX{i} = X;

end

parts = cat(1,XX{:});

for i = 1:k
    mom(:,i) = sum(XX{i},1);
end