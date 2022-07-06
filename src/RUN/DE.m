function [Convergence_curve,Best_Cost,Best_X]=DE(nP,Max_It,lb,ub,dim,fobj,F,Cr)


% Initialization
Convergence_curve=zeros(Max_It,1);

X=zeros(nP,dim);
Cost=zeros(nP,1);

for i=1:nP
    
    X(i,:)=unifrnd(lb,ub,1,dim);
    Cost(i)=fobj(X(i,:));
        
end

[Best_Cost,ind] = min(Cost);
Best_X = X(ind,:);

% Main Loop

for it=1:Max_It    
    
    for i=1:nP
                
        Rand_ind=randperm(nP);
        
        Rand_ind(Rand_ind==i)=[];
        
        a = Rand_ind(1);
        b = Rand_ind(2);
        c = Rand_ind(3);

        % Mutation
        y = X(a,:)+F.*(X(b,:)-X(c,:));

        % Crossover
        z=zeros(1,dim);
        j0=randi([1 dim]);
        for j=1:dim
            if j==j0 || rand<=Cr
                z(j)=y(j);
            else
                z(j)=X(i,j);
            end
        end
        
        New_X = min(max(z,lb),ub);
        New_Cost = fobj(New_X);

        if New_Cost<Cost(i)
            X(i,:) = New_X;
            Cost(i) = New_Cost;            
            if Cost(i)<Best_Cost
               Best_X = X(i,:);
               Best_Cost = Cost(i);
            end
        end
        
    end

    Convergence_curve(it)=Best_Cost;
    
    % Show Information
    disp(['Iteration ' num2str(it) ': BestCost = ' num2str(Convergence_curve(it))]);
    
end
    



