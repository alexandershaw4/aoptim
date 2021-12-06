function e = thr_dist(Y,y,w)
%

%Y = detrend(Y);
%y = detrend(y);

%pf=polyfit(w,Y,3);
%pfw = polyval(pf,w);
%pfw = pfw - min(pfw);

%Y = Y - pfw(:);
%y = y - pfw(:);

[~,iY] = sort(Y,'descend');
[~,iy] = sort(y,'descend');

Q  = sqrt( (iY.^2) - (iy.^2) );
%N  = round(length(Q)*.2);

%e  = sum(Q(1:N));
e  = sum(Q);



