function errplot(mu,var,type)

w = 1:length(mu);

if nargin < 3 || isempty(type)
    type = 'scatter';
end


switch lower(type)
    case 'scatter'
        s = scatter(w,mu,100,'filled'); hold on;
        s.CData = [1 0 1];
    case 'line';
        s = line(w,mu,'color',[1 0 1]);hold on;
end


for i = 1:length(mu)
    l(1) = line([w(i) w(i)],[mu(i)-var(i) mu(i)+var(i)]);
    l(2) = line([w(i)-.2 w(i)+.2],[mu(i)-var(i) mu(i)-var(i)]);
    l(3) = line([w(i)-.2 w(i)+.2],[mu(i)+var(i) mu(i)+var(i)]);
    
    l(1).Color = 'k';
    l(2).Color = 'k';
    l(3).Color = 'k';
end


hold off;
