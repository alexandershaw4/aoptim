function y = quickgd(f,x)

j = dfdx(f,x);

iter = true;

e = f(x);

n = 0;

while iter
    de = f(x - (1/8*j) );
    
    if de < e
        e = de;
        x = x - (1/8*j);
        n = n + 1;
    else
        iter = false;
    end
end

fprintf('converged after %d iterations\n',n);
y = x;

end

function j = dfdx(f,x)

fx = f(x);
h  = 1/8;

for i  = 1:length(x)
    
    x0    = x;
    x0(i) = x0(i) + h;
    
    j(i)  = ( f(x0) - fx ) / h;

end 
end