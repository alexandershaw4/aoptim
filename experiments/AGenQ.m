function Q = AGenQ(x)

% Peak points as regions of importance
x     = x ./ max(x);
[v,I] = findpeaks(x);

x(I)  = 8;

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