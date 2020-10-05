function [GO] = genGO(Go,trunc)
N=length(Go);

if ~exist('trunc')
    trunc=0;
end

nck0=nchoosek(N,0);
nck1=nchoosek(N,1);
try
    nck2=nchoosek(N,2);
catch
    nck2=0;
end
try
    nck3=nchoosek(N,3);
catch
    nck3=0;
end
%GO=zeros(N*(N-1)/2,N);
GO=zeros(nck2,nck1);

% 
% for i=1:N-1
% 
% n=i;
% 
% x=N*(N-1)/2+1-i*(i+1)/2;
% y=N-i;
% 
% Go=[ones(1,n)',eye(n)];
% [r,c]=size(Go);
% GO(x:x-1+r,y:y+c-1)=Go;
% %genGo_block(n)
% 
% end
GO=[];
for i=N-trunc:-1:2
    
GO_temp=[ones(i-1,1),eye(i-1)];
GO_temp=[Go(N+2-i:N)',Go(N+1-i)*eye(i-1)];

GO=flip((catpad(1,flip(GO,2),flip(GO_temp,2))),2);

end

GO(isnan(GO))=0;


end