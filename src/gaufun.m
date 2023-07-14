classdef gaufun < handle
    % "pca/svd under the assumption of Gaussians" or "gaussian process pca"
    % 
    % essentially makes the assumption that each point in the input vector 
    % can be considered the mean of a fixed-width Gaussian (thus the input
    % is a GMM), turning the vector to a matrix, to which pca can be
    % applied.
    % 
    % -the vector to matrix function gaufun.AGenQn implements
    %   f = @(x,n) diag(ones(N-abs(x),1),x)/n; s.t. f'*f = f*f'
    %
    % -the pca function GaussPCA is an svd of the smooth matrix generated
    % by the above.
    %
    % -the search function SearchGaussPCA repeatedly calls the above for
    % models with between 1:N components and returns the model closest to
    % the input assessed with distance-correlation
    %
    % [J] = gaufun.SearchGaussPCA(X,N)
    %
    % AS

    properties

    X
    J
    Q

    end

    methods(Static) % making them static means the subfuns are callable 
                    % without instantiating a gaufun object

        
        function [J,win] = SearchGaussPCA(X,N)
            
            % transpose if req
            if any(size(X)==1) && size(X,1)==1
                X = X';
            end

            % compute N-dim PCs
            for i = 1:N
                J{i} = gaufun.GaussPCA(X,(i));
            end

            % compare
            C = cat(1,J{:});
            for i = 1:N
                %r(i) = corr(X(:),J{i}(:)).^2;
                C = real(C);
                X = real(X);
                r(i) = distcorr(C(i,:)',X);

            end
            
            [~,win] = min(r);
                                   
            J = J{win};
            
            
        end

        function [J,QM,v] = GaussPCA(X,N)

            if nargin < 2 || isempty(N)
                N = 1;
            end

            for i = 1:size(X,2)

                QM = gaufun.AGenQn(X(:,i),8);
                %QM = atcm.fun.VtoGauss(X(:,i));
                %QM = gaufun.QtoGauss(X(:,i),2);
                %QM=(QM+QM')/2;
                
                [u,s,v] = svd((QM));
                J(i,:) = sum( QM*v(:,1:N), 2);
                

            end

            J = abs(J);

        end

        function [Q,GL] = AGenQn(x,n)
            % Convert vector into Gaussian process, i.e. a smoothed symmetric matrix form.
            %
            % Q = AGenQn(x,n)
            %
            % Implements function
            %       f = @(x,n) diag(ones(N-abs(x),1),x)/n;
            % for the sequence
            %       M = f(0,1) + f(1,2) + f(-1,2) + f(2,4) + f(-2,4) + f(3,8) + f(-3,8) ...
            %
            % AS2022

            if nargin < 2 || isempty(n)
                n = 3;
            end

            N = length(x);

            f = @(x,n) diag(ones(N-abs(x),1),x)/n;

            % a la, M = f(0,1) + f(1,2) + f(-1,2) + f(2,4) + f(-2,4) + f(3,8) + f(-3,8);

            M = f(0,1);

            for i  = 1:n
                s  = cumprod(repmat(2,[1 i]));
                M  = M + f(i,s(end)) + f(-i,s(end));
            end

            % linear model
            %Q = smooth2(denan(M.\diag(x)),4);
            Q = smooth2(M*diag(x),4);

            
            
    
            if nargout == 2
                %Q  = cdist(x,x) .* Q;
                A  = Q .* ~eye(length(Q));
                N  = size(A,1);
                GL = speye(N,N) + (A - spdiags(sum(A,2),0,N,N))/4;
            end
        end

        function [G,b,GL] = QtoGauss(Q,w,model)
            % given a vector that has converted to a smoothed symmetrical matrix,
            % perform explicit conversion to Gaussians
            %
            % e.g.
            %      x = [-10:1:10];           % a vector of data
            %      Q = atcm.fun.AGenQn(x,4); % convert to smoothed matrix
            %      Q = .5*(Q+Q');            % ensure symmetric
            %      G = atcm.fun.QtoGauss(Q); % convert to Gaussian series
            %
            % AS22

            if nargin < 3 || isempty(model)
                model = 'Gauss';
            end

            if nargin < 2 || isempty(w)
                w = 4;
            end

            if isvector(Q)
                try
                    Q = gaufun.AGenQn(Q,4);
                catch
                    Q = gaufun.AGenQn(Q,1);
                end
            end

            G = Q*0;
            x = 1:length(Q);

            for i = 1:length(Q)

                t = Q(:,i);

                [v,I] = max(t);

                if length(w) == length(Q)
                    G(:,i) = gaufun.makef(x,I-1,v,w(i),model);
                else
                    G(:,i) = gaufun.makef(x,I-1,v,w,model);
                end

            end

            if nargout == 2
                % Reduce if requested
                b = atcm.fun.lsqnonneg(G,diag(Q));
            else
                b = [];
            end

            if nargout == 3
                Q  = G;
                A  = Q .* ~eye(length(Q));
                N  = size(A,1);
                GL = speye(N,N) + (A - spdiags(sum(A,2),0,N,N))/4;
            end

        end

        function X = makef(w,Fq,Amp,Wid,model)
            % Generate a Gaussian/Cauchy/Lapalce/Gamma distribution / mixture, e.g.
            %
            %   atcm.fun.makef(w,Fq,Amp,Wid,model)
            %
            % e.g.
            % w = 1:100
            % S = afit.makef(w,[10 20 50 70],[10 8 5 3],[1 2 5 3],'gaussian');
            % figure;plot(w,S)
            %
            % AS2019

            if nargin < 5 || isempty(model) % 'gaussian or cauchy
                model = 'gaussian';
            else
                model = model;
            end

            if isstruct(Fq)
                Amp = Fq.Amp;
                Wid = Fq.Wid;
                Fq  = Fq.Freq;
            end

            if length(Fq) > 1
                for i = 1:length(Fq)
                    try
                        X0 = X0 + gaufun.makef(w,Fq(i),Amp(i),Wid(i),model);
                    catch
                        X0 =      gaufun.makef(w,Fq(i),Amp(i),Wid(i),model);
                    end
                    %X0(i,:) = afit.makef(w,Fq(i),Amp(i),Wid(i));

                end
                %X0 = max(X0);
                X  = X0;
                return;
            end


            try Wid ; catch Wid = 2; end
            try Amp ; catch Amp = 2; end

            % offset negative values in w
            mw  = min(w);
            X   = 0*w;
            f   = gaufun.findthenearest(Fq,w);

            try
                f   = f(1);
            catch
                f   = Fq(1);
            end

            w   = w - mw;
            switch model
                case {'Gauss' 'gauss' 'gaussian'}
                    X   = X + Amp * exp( -(w-f).^2 / (2*(2*Wid)^2) );
                case 'cauchy'
                    X   = X + Amp./(1+((w-f)/Wid).^2);
                case 'laplace'
                    X    = X + Amp * ( 1/(sqrt(2)*Wid)*exp(-sqrt(2)*abs(w-f)/Wid) );
                case 'gamma'
                    X    = X + Amp * gampdf(w,f);
            end
            w   = w + mw;

        end

        function [r,c,V] = findthenearest(srchvalue,srcharray,bias)
            %
            %
            % By Tom Benson (2002)
            % University College London
            % t.benson@ucl.ac.uk

            if nargin<2
                error('Need two inputs: Search value and search array')
            elseif nargin<3
                bias = 0;
            end

            % find the differences
            srcharray = srcharray-srchvalue;

            if bias == -1   % only choose values <= to the search value

                srcharray(srcharray>0) =inf;

            elseif bias == 1  % only choose values >= to the search value

                srcharray(srcharray<0) =inf;

            end

            % give the correct output
            if nargout==1 | nargout==0

                if all(isinf(srcharray(:)))
                    r = [];
                else
                    r = find(abs(srcharray)==min(abs(srcharray(:))));
                end

            elseif nargout>1
                if all(isinf(srcharray(:)))
                    r = [];c=[];
                else
                    [r,c] = find(abs(srcharray)==min(abs(srcharray(:))));
                end

                if nargout==3
                    V = srcharray(r,c)+srchvalue;
                end
            end

        end



    end



end