function [X,F,CV,H,par] = reductionoptimise(fun,x0,V,y)

ip = find(V);

if nargin < 4 || isempty(y)
    y = 0;
end

for i = 1:24

    fprintf('DMD_OPT: Iteration %d/%d\n',i,24);
    
    % compute partial numerical derivatives
    J = jaco_mimo(fun,x0,V,0,2,1);
    J = cat(1,J{:});
    J = denan(J);

    % DMD to get Koopman operator
    [~,E] = atcm.fun.dmd(J',rank(J),1);

    % new reduced space values
    rX = ones(size(E,1),1);
    rV = ones(size(E,1),1)/8;

    XX = spm_svd(diag(V));
    f  = @(p) spm_vec(fun( x0 - (XX*(p'*E)') ));

    % now optimise with AO
    op = AO('options');

    op.fun = f;
    op.x0  = rX;
    op.V   = rV;
    op.y   = y;

    op.ismimo=1;
    op.isGaussNewtonReg = 1;
    op.criterion = -inf;

    [X,F,CV,H,par] = AO(op);
    
    % update
    x0 = x0 - (XX*(X'*E)');
    
end

par.E  = E;
par.XX = XX;