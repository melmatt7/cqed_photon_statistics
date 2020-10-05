function [J3] = genJ3(jvec)

nck2=length(jvec);
nck1=round(sqrt(2*nck2+0.25)+0.5);
N=nck1;
B=tril(ones(nck1),-1);
B(B==1)=jvec;
J=(B+transpose(B));

if N<3
    J3=[];
    return
elseif N==3
    J3=0;
    return
end

%gen uptri, then block beneath, etc

jtri=tril(ones(nck1-1));
jtri(jtri==1)=jvec;
J3=[];

for j=1:N-1
    J2=[];
    
    jcol=jtri(:,j);
    
    for i=2+j-1:N-2
        jtri_sub=jtri(i:end,i:end);

        %j0=j(1,1);
        j1=jtri_sub(1:end,1);
        j2=jtri_sub(2:end,2:end);

        jt=[];
        for k=1:size(j2,1)
        jt_temp=[j1(k+1:end),j1(k)*eye(size(j1,1)-k)];

        %jt=catpad(1,jt,jt_temp);
        jt=flip((catpad(1,flip(jt,2),flip(jt_temp,2))),2);
        end
        jt=catpad(1,j2,jt);

        %J2=catpad(2,J2,jt);
        J2=flip((catpad(2,flip(J2,1),flip(jt,1))),1);
        J2(isnan(J2))=0;
    end
    %now add square matrix beneath it
    
    Jbox=[];
    for i=1:N-3
        Jt=[];
        jt=[];
        for k=i+j:size(jcol-1,1)
            jt_temp=[jcol(k+1:end),jcol(k)*eye(size(jcol-1,1)-k)];

            %jt=catpad(1,jt,jt_temp);
            jt=flip((catpad(1,flip(jt,2),flip(jt_temp,2))),2);
            jt(isnan(jt))=0;
        end
        if size(jt,1)>0
            Jt=[jt,jcol(i+j-1)*eye(size(jt,1))];
        end
        Jbox=flip((catpad(1,flip(Jbox,2),flip(Jt,2))),2);
    end
    Jbox(isnan(Jbox))=0;
    
    for k=1+j:size(jcol-1,1)
        jt_temp=[jcol(k+1:end),jcol(k)*eye(size(jcol-1,1)-k)];

        %jt=catpad(1,jt,jt_temp);
        jt=flip((catpad(1,flip(jt,2),flip(jt_temp,2))),2);
        jt(isnan(jt))=0;
    end
    
    J2=catpad(2,J2,0);
    J3_sub=flip((catpad(1,flip(J2,2),flip(Jbox,2))),2);
    J3_sub(isnan(J3_sub))=0;
    
    J3=flip((catpad(2,flip(J3,1),flip(J3_sub,1))),1);
    %J3=[J3,J3_sub];
end

J3=catpad(1,0,J3);
J3(isnan(J3))=0;
J3=J3(:,1:end-1);
J3=(J3+transpose(J3));

end