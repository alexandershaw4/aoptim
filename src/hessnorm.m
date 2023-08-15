function [H] = hessnorm(H)

% check conditioning 
if cond(H) > 10
    [us,ss,vs] = svd(H); ss = diag(ss);
    ss = rescale(ss,max(ss)/10,max(ss));
    H = us*diag(ss)*vs';
end
            

%dH = H ./ (2*max(H));
%Hx = dH+dH';
%n  = 0;

%while cond(Hx) < cond(H)
%    H  = Hx;
%    dH = H ./ (2*max(H));
%    Hx = dH+dH';
%    n  = n + 1;
%end

end