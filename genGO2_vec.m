function [GO2] = genGO_2(Go)
N=length(Go);

if N <3
    GO2 = [];
    return
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
%GO2=zeros(nck2,nck3);
GO2=[];
for i=N:-1:3
GO2_temp=[genGO_vec(Go,N+1-i),Go(N+1-i)*eye(nchoosek(i-1,2))]

GO2=flip((catpad(1,flip(GO2,2),flip(GO2_temp,2))),2)

end

GO2(isnan(GO2))=0;

end