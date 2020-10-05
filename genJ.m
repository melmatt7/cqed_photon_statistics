function [J] = genJ2(jvec)

nck2=length(jvec);
nck1=round(sqrt(2*nck2+0.25)+0.5);
B=tril(ones(nck1),-1);
B(B==1)=jvec;
J=(B+transpose(B));

end


