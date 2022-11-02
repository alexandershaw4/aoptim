function covQ = makeposdef(covQ)
% Ensure a covariance matrix is positive, semidefinite
%
%   Q = makeposdef(Q)
%
% AS2022

tol = 1e-3;

if size(covQ,1) == size(covQ,2)
    covQ = (covQ + covQ')/2;
    
    % regularise
    covQ(isnan(covQ))=tol;
    covQ(isinf(covQ))=tol;
    
    % make sure its positive semidefinite
    lbdmin = min(eig(covQ));
    boost = 2;
    covQ = covQ + ( boost * max(-lbdmin,0)*eye(size(covQ)) );

else
    
    Q = covQ;
    covQ = sqrt(covQ * covQ');
    
    % regularise
    covQ(isnan(covQ))=tol;
    covQ(isinf(covQ))=tol;
    
    % make sure its positive semidefinite
    lbdmin = min(eig(covQ));
    boost = 2;
    covQ = Q + ( boost * max(-lbdmin,0)*eye(size(Q)) );
    
end