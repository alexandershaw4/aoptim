function [Q,GL] = AGenQ(x)
% Returns an auto smoothing matrix Q for the elements of x, and its
% graph Laplacian
%
% [Q,GL] = AGenQ(x)
%
% AS
 
% Q = x + (ux) + (lx);

% % Peak points as regions of importance
% x     = x ./ max(x);
% [v,I] = findpeaks(x);
% x(I)  = 8;
 
% Get features
a  = diag( x );

% 1sts
at = a(:,2:end)/2;
at(:,end+1) = 0;

ab = a(2:end,:)/2;
ab(end+1,:)=0;

% 2nds
at2 = a(:,3:end)/4;
at2(:,end+2) = 0;

ab2 = a(3:end,:)/4;
ab2(end+2,:)=0;

% 3rds
at3 = a(:,4:end)/8;
at3(:,end+3) = 0;

ab3 = a(4:end,:)/8;
ab3(end+3,:)=0;


% Compose Q
Q = a + at  + ab  + ... 
        at2 + ab2 + ...
        at3 + ab3 ;
    
Q = smooth2(Q,4);

if nargout == 2
    %Q  = cdist(x,x) .* Q;
    A  = Q .* ~eye(length(Q));
    N  = size(A,1);
    GL = speye(N,N) + (A - spdiags(sum(A,2),0,N,N))/4;
end
