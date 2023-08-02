function [H,n] = hessnorm(H)

dH = H ./ (2*max(H));
Hx = dH+dH';
n  = 0;

while cond(Hx) < cond(H)
    H  = Hx;
    dH = H ./ (2*max(H));
    Hx = dH+dH';
    n  = n + 1;
end

end