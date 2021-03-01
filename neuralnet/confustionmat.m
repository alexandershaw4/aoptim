function [M,T] = confustionmat(Yy)

y = Yy(:,1); x = Yy(:,2); n = length(x);

TP = 100*sum( y &  x )./n;
TN = 100*sum(~y & ~x )./n;
FP = 100*sum(~y &  x )./n;
FN = 100*sum( y & ~x )./n;

M = [TP FN;
     FP TN];
 
T = array2table(M(:)','VariableNames',{'TP' 'FP' 'FN' 'TN'});
T
