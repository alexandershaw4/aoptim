function T = predictive(x)
% x is a nx2 binary matrix consisting:
%
% truths in x(:,1)
% SVM / model predictions in x(:,2)
%
% Output is a structure containing TP, TN, Sens(%), Spec(%), ...
% PPV(%) and NPV(%)
%
% AS

%x = double(logical(x));

% Make sure truths are sorted 0 then 1s
[~,I] = sort(x(:,1));
x = x(I,:);

s = @sum;
l = @length;
f = @find;
a = @abs;
x = double(x);

% seperate
truth = x(:,1);
pred  = x(:,2);

% group indices
n     = find(truth==1);
n     = n(1);
n     = 1:n-1;
if isempty(n); n = 0; end
p     = n(end)+1:length(truth);


% truths
T.TP = s( truth(p) &  pred(p)); % true pos
T.TN = s(~truth(n) & ~pred(n)); % true neg

T.FP = a(l(f(truth(n))) - l(f(pred(n)))); % false pos
T.FN = a(l(f(truth(p))) - l(f(pred(p)))); % false neg


% predictive powers
T.Sensitivity = 100*(T.TP / (T.TP + T.FN));
T.Specificity = 100*(T.TN / (T.TN + T.FP));

T.PPV         = 100*(T.TP / (T.TP + T.FP)); % positive predictive power
T.NPV         = 100*(T.TN / (T.TN + T.FN)); % negative predictive power



% NOTES
%----------------------------------------------------------------------
% PPV = If your test comes back +ve, how many will really have disease
% NPV = If your test comes back -ve, how many truely do not have diease

