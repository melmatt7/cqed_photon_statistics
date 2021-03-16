clear
%close all
%% Initialization
tic

set(groot,'defaultfigureposition',[400 250 900 700])
showfig1=1;
showfig2=0;
showfig3=0; %log
showfig4=0; %time

showg2d=0;
Hefft=0;

g3fig=0;
g3time=0;

mu=0;
sigma=0;

tnum=2;
t_end=20;

%plotting parameters
s=0.2;
lim1=-80;
lim2=80;
wnum=5000;

minval=6.66;
time=0;

n=50;
N=n;

k=43;   %  4.3 - Q~100,000 , 43 - Q~10,000  %cavity linewidth
k1=k/2;
k2=k/2;
g=.043;   % emitter linewidth
Gc=0;   %additional loss channel
wc=0;  %cavity detuning
we=0;   %emitter detuning
   


go=.32;  %rabi frequency


%% Start of Loop
wc_list=[0 5 10];
we_list=[0 3 10];
n_list = [10, 100, 500];
for z = 1:length(we_list)
    y=1;
    N=n_list(z);
    
    Go=go*ones(1,N);
    G=g*ones(1,N);
    K=k*ones(1,N);
    %We=normrnd(mu,sigma,1,N);
    We=we_list(1)*ones(1,N);
    w=linspace(lim1,lim2,wnum);
    
    J=0;%dipole-dipole coupling
    jvec=J.*ones(1,N*(N-1)/2);

    %values used to dimension the higher excitation hamiltonians
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

    %% creation/annihilation operators
    gnd=1;
    a1=[1,zeros(1,N)];

    a2=[eye(N+1),repmat(zeros(1,N+1)',1,nck2)];
    a2(1)=a2(1)*sqrt(2);
% 
%     a3=zeros(nck0+nck1+nck2,nck0+nck1+nck2+nck3);
%     a3(1,1)=sqrt(3);
%     a3(2:(nck1+nck0),2:(nck1+nck0))=eye(nck1)*(sqrt(2));
%     a3(nck0+nck1+1:nck0+nck1+nck2,nck0+nck1+1:nck0+nck1+nck2)=eye(nck2);


    %% Zeroth excitation subspace
    Heff0=0;

    %% First Excitation Subspace
    Heff1=zeros(N+1);
    %cavity loss
    Heff1(1,1)=wc_list(y)-1i*k/2; %+1iw comes from the diagonal contribution later
    if N>0
        %atomic coupling
        Heff1(1,2:N+1)=Go;
        Heff1(2:N+1,1)=Go;
    end
    %spontaneous emission
    if N>0
        Heff1(2:N+1,2:N+1)=Heff1(2:N+1,2:N+1)+(diag(We)-1i*diag(G)/2);
    end

    J=genJ(jvec);
    Heff1(2:nck0+nck1,2:nck0+nck1)=Heff1(2:nck0+nck1,2:nck0+nck1)+J;


    %unsorted phi1
    [phi1,lambda1]=eig(Heff1);
    lambda1m=lambda1;
    lambda1=diag(lambda1);
    
    norm=sqrt((sum(phi1.^2)));
    phi1n=[];
    for i=1:size(phi1,2)
        phi_temp=phi1(:,i)/norm(i);
        phi1n=[phi1n,phi_temp];
    end
    phi1v=inv(phi1);
    phi1*lambda1m*phi1v;

    %% Second Excitation Subspace
    if showfig2==1
        Heff2=zeros(1+N*(N+1)/2);
        %cavity loss
        Heff2(1,1)=2*(wc_list(y)-1i*k/2); %+1iw comes from the diagonal contribution later
        %atom-cavity coupling
        Heff2(1,2:N+1)=sqrt(2)*Go;
        Heff2(2:N+1,1)=sqrt(2)*Go;

        % first set of diagonals
        if N>=1
            Heff2(2:N+1,2:N+1)=(eye(N)*wc_list(y)-1i*diag(K)/2)+(diag(We)-1i*diag(G)/2);
        end
        %second set of diagonals
        if N>=1
            sum(nchoosek((We-1i*ones(1,N)*g/2),2),2);
            Heff2(N+2:end,N+2:end)=diag(sum(nchoosek((We-1i*ones(1,N)*g/2),2),2));
        end

        %Dipole-dipole coupling
        J=genJ(jvec);
        Heff2(nck0+1:nck0+nck1,nck0+1:nck0+nck1)=Heff2(nck0+1:nck0+nck1,nck0+1:nck0+nck1)+J;
        J2=genJ2(jvec);
        Heff2(nck0+nck1+1:nck0+nck1+nck2,nck0+nck1+1:nck0+nck1+nck2)=Heff2(nck0+nck1+1:nck0+nck1+nck2,nck0+nck1+1:nck0+nck1+nck2)+J2;
        
        %Go
        GO=genGO_vec(Go);
        [r,c]=size(GO);

        Heff2(N+2:N+2+r-1,2:2+c-1)=Heff2(N+2:N+2+r-1,2:2+c-1)+GO;
        Heff2(2:2+c-1,N+2:N+2+r-1)=Heff2(2:2+c-1,N+2:N+2+r-1)+GO';

        %%unsorted phi2
        [phi2,lambda2]=eig(Heff2);
        lambda2m=lambda2;
        lambda2=diag(lambda2);
        norm2=sqrt((sum(phi2.^2)));
        phi2n=[];
        for i=1:size(phi2,2)
            phi_temp2=phi2(:,i)/norm2(i);
            phi2n=[phi2n,phi_temp2];
        end
        phi2v=inv(phi2);
        phi2n;
        phi2*lambda2m*phi2v;
    end

    %% Third Excitation Subspace
    if g3fig==1
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
        %start of each diagonal
        i1=1;
        i2=1+nck0;
        i3=1+nck0+nck1;
        i4=1+nck0+nck1+nck2;

        dim=nck0+nck1+nck2+nck3;
        Heff3=zeros(dim);
        %first diagonal
        Heff3(1,1)=3*(wc_list(y)-1i*k/2); 
        %second diagonal
        Heff3(i2:i3-1,i2:i3-1)=2*(eye(N)*wc_list(y)-1i*diag(K)/2);
        Heff3(i2:i3-1,i2:i3-1)=Heff3(i2:i3-1,i2:i3-1)+(diag(We)-1i*diag(G)/2);
        %third diagonal
        Heff3(i3:i4-1,i3:i4-1)=(eye(i4-i3).*(wc_list(y)-1i*k/2));
        Heff3(i3:i4-1,i3:i4-1)=Heff3(i3:i4-1,i3:i4-1)+diag(sum(nchoosek((We-1i*ones(1,N)*g/2),2),2));  %combnk sometimes flips the order and is thus unreliable
        %fourth diagonal
        Heff3(i4:end,i4:end)=diag(sum(nchoosek((We-1i*ones(1,N)*g/2),3),2));

        % 0-1 coupling sqrt(3)go
        Heff3(i1,i2:(i3-1))=sqrt(3)*Go;
        Heff3(i2:(i3-1),i1)=sqrt(3)*Go;
        % 1-2 coupling sqrt(2)go
        Heff3(i3:i4-1,i2:i3-1)=sqrt(2)*genGO_vec(Go);
        Heff3(i2:i3-1,i3:i4-1)=sqrt(2)*genGO_vec(Go)';
        % 2-3 coupling go
        Heff3(i4:end,i3:i4-1)=genGO2_vec(Go);
        Heff3(i3:i4-1,i4:end)=genGO2_vec(Go)';

        % Dipole-dipole coupling
        J=genJ(jvec);
        Heff3(nck0+1:nck0+nck1,nck0+1:nck0+nck1)=Heff3(nck0+1:nck0+nck1,nck0+1:nck0+nck1)+J;
        J2=genJ2(jvec);
        Heff3(nck0+nck1+1:nck0+nck1+nck2,nck0+nck1+1:nck0+nck1+nck2)=Heff3(nck0+nck1+1:nck0+nck1+nck2,nck0+nck1+1:nck0+nck1+nck2)+J2;
        J3=genJ3(jvec);
        Heff3(nck0+nck1+nck2+1:nck0+nck1+nck2+nck3,nck0+nck1+nck2+1:nck0+nck1+nck2+nck3)=Heff3(nck0+nck1+nck2+1:nck0+nck1+nck2+nck3,nck0+nck1+nck2+1:nck0+nck1+nck2+nck3)+J3;

        %%unsorted phi3
        [phi3,lambda3]=eig(Heff3);
        lambda3m=lambda3;
        lambda3=diag(lambda3);
        norm3=sqrt((sum(phi3.^2)));
        phi3n=[];
        for i=1:size(phi3,2)
            phi_temp3=phi3(:,i)/norm3(i);
            phi3n=[phi3n,phi_temp3];
        end
        phi3v=inv(phi3);
    end

    %% Rest

%     N1=length(lambda1);
%     %N2=length(lambda2);
% 
%     t2=0;
% 
% 
%     % Heff01=vertcat([Heff0,a1],[a1',Heff1]);
%     % Heff12=vertcat([Heff1,a2],[a2',Heff2]);
%     % Heff=vertcat( [0,1,zeros(1,size(Heff12,2)-1)], [ [1,zeros(1,size(Heff12,1)-1)]',Heff12]);
% 
%     t2=0;
%     for i=1:N1
%         t2=t2+phi1(1,i).*(phi1(1,i))./(lambda1(i)-w);
%     end
% 
     winc=(lim2-lim1)/wnum;
    for i =1:wnum
        w=lim1+i*winc;
        %w=0;
        w*ones(1,N+1);
        diag(w*ones(1,N+1));
        Dnum=lambda1m-diag(w*ones(1,N+1));
        D=Dnum.^(-1);
        D;

        %pp1=gnd'*a1'*phi1;
        %pp1v=phi1v*g;
        t2(i)=1i*gnd'*a1*phi1*D*phi1.'*a1'*gnd;
        W(i)=w;
    end

    %% Frequency dependence g1
    if showfig1==1
        t_inc=t_end/tnum;
        for i =1:wnum
            w=lim1+i*winc;
            %w=2.9;
            %w=0;
            D1num=lambda1m-diag(w*ones(1,N+1));
            D1=diag(diag(D1num).^(-1));
            %Transmission
%             t(i)=1i*gnd'*a1*phi1n*D1*phi1n.'*a1'*gnd; 
            t(i)=1i*gnd'*a1*phi1*D1*phi1v*a1'*gnd;
        end
    end

    %% Frequency Dependence g2 and g3
    % w dependence
    if showfig2==1
        t_inc=t_end/tnum;
        for i =1:wnum
            w=lim1+i*winc;
            %w=2.9;
            %w=0;
            lambda1m;
            
            D1num=lambda1m-diag(w*ones(1,N+1));
            D1=diag(diag(D1num).^(-1));   
            D1;
            D2num=lambda2m-diag(2*w*ones(1,N*(N+1)/2+1));
            D2=diag(diag(D2num).^(-1)); 
            if g3fig==1        
                D3num=lambda3m-diag(3*w*ones(1,nck0+nck1+nck2+nck3));
                D3=diag(diag(D3num).^(-1));
            end
            %Transmission
            t(i)=1i*gnd'*a1*phi1n*D1*phi1n.'*a1'*gnd; 
            t(i)=1i*gnd'*a1*phi1*D1*phi1v*a1'*gnd; 

            T=k1*k2*t(i).*conj(t(i));
            T(i)=T;

            %second order correlation
            j = phi1v*a1'*gnd*gnd'*a1*phi1*D1*phi1v*a1'*a1;
            gw1_diag=diag(phi1v*a1'*gnd*gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1);

            W(i)=w;
            fw1_diag=diag(phi1v*a2*phi2*D2*phi2v*a2'*phi1*D1*phi1v*a1'*gnd*gnd'*a1*phi1);
            size(fw1_diag);
            size(gw1_diag);
            b = (fw1_diag.')*exp(-1i*(lambda1-w)*(time));
            a = gw1_diag.'*(ones(1,size(phi1,1)))';
            fw2(i)=(fw1_diag.')*exp(-1i*(lambda1-w)*(time)) + gw1_diag.'*(ones(1,size(phi1,1))'-exp(-1i*(lambda1-w)*(time)));

            %third order
            if g3fig==1
                g3w(i)=gnd'*a1*phi1*phi1v*a2*phi2*phi2v*a3*phi3*D3*phi3v*a3'*phi2*D2*phi2v*a2'*phi1*D1*phi1v*a1'*gnd;    
            end
        end
    end

    %2d second order correlation plot with detuning and time
    if showg2d==1
        figure(2)
        g2wt(g2wt>3)=3;
        [X,Y] = meshgrid(W-wc_list(y),time_vec);
        figure(1)
        contourf(X,Y,g2wt)
        caxis([0 3])
        colorbar
        figure(2)
        meshz(X,Y,g2wt)
        %plot(W-wc,g2wt(1,:))
        %ylim([0 5])

        figure(3)
        hold off
        T=k1*k2.*t.*conj(t);
        g2=k1^2*k2^2.*g2w.*conj(g2w)./(T.^2);
        plot(W-wc_list(y),T)
        hold on
        plot(W-wc_list(y),g2)
        ylim([0 5])
        if g3fig==1
            g3=k1^3*k2^3.*g3w.*conj(g3w)./(T.^3);
            plot(W-wc_list(y),g3)
        end
        legend('Transmission','g2(0)','g3(0)')
    end


    % figure(3)
    % splot=surf(g2wt,'FaceAlpha',1)
    % splot.FaceColor='flat';
    % splot.EdgeColor='flat';
    %% Time Dependence

    t_end=40;
    tinc=0.01;
    tnum=t_end/tinc+1;
    Time=0:tinc:t_end;

    if showfig4==1
        % Transmission
        for i =1:tnum
            time=0+i*tinc;
            w=wc_list(y)+4.5;  %strong coupling
            %w=wc+0.895;  %weak coupling
            %w=wc+0.11;  %weak coupling
            %w=wc+minval;

            D1num=lambda1m-diag(w*ones(1,N+1));
            D1=diag(diag(D1num).^(-1));   
            D2num=lambda2m-diag(2*w*ones(1,N*(N+1)/2+1));
            D2=diag(diag(D2num).^(-1));   

            t_t(i)=1i*gnd'*a1*phi1n*D1*phi1n.'*a1'*gnd;
            t_t(i)=1i*gnd'*a1*phi1*D1*phi1v*a1'*gnd;

            T_t_trans(i)=k1*k2*t_t(i).*conj(t_t(i));

            gw(i)=gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1*phi1v*a1'*gnd;
            %fw(i)=gnd'*a1*phi1*phi1v*a2*phi2*D2*phi2v*a2'*phi1*D1*phi1v*a1'*gnd;
            fw(i)=gnd'*a1*a2*phi2*D2*phi2v*a2'*phi1*D1*phi1v*a1'*gnd;

            gw1_diag=diag(phi1v*a1'*gnd*gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1);
            gw1_diag=diag(phi1v*a1'*gnd*gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1);

            fw1_diag=diag(phi1v*a2*phi2*D2*phi2v*a2'*phi1*D1*phi1v*a1'*gnd*gnd'*a1*phi1);
            fw=gw1_diag.'*(ones(1,size(phi1,1))') + (fw1_diag.'-gw1_diag.')*exp(-1i*(lambda1-w)*(time));
            fw2_t(i)=fw;      
            %fw2_t(i)=gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1*phi1v*a1'*gnd;

            if g3time==1        
                D3num=lambda3m-diag(3*w*ones(1,nck0+nck1+nck2+nck3));
                D3=diag(diag(D3num).^(-1));

                c1=gnd'*a1*phi1*D1*phi1v*a2*D2*phi2v*a3*phi3*D3*phi3v*a3'*phi2*phi2v*a2'*phi1*phi1v*a1'*gnd;
                c1_diag=diag(phi1v*a1'*gnd*gnd'*a1*phi1*D1*phi1v*a2*D2*phi2v*a3*phi3*D3*phi3v*a3'*phi2*phi2v*a2'*phi1);
                c2=gnd'*a1*phi1*D1*phi1v*a2*D2*phi2v*a3*phi3*D3*phi3v*a3'*phi2*phi2v*a2'*phi1*phi1v*a1'*gnd;
                c2=diag(D1*phi1v*a2*D2*phi2v*a3*phi3*D3*phi3v*a3'*phi2*phi2v*a2'*phi1*phi1v*a1'*gnd*gnd'*a1*phi1);
            end
        end
    end

    if showfig4==1
        % Reflection
        for i =1:tnum
            time=0+i*tinc;
            w=wc_list(y)+0; % strong coupling
            w=wc_list(y)+0;  % weak coupling

            D1num=lambda1m-diag(w*ones(1,N+1));
            D1=diag(diag(D1num).^(-1));   
            D2num=lambda2m-diag(2*w*ones(1,N*(N+1)/2+1));
            D2=diag(diag(D2num).^(-1));   

            t_t(i)=1i*gnd'*a1*phi1n*D1*phi1n.'*a1'*gnd;
            t_t(i)=1i*gnd'*a1*phi1*D1*phi1v*a1'*gnd;

            %T_t_ref(i)=k1*k2*t_t(i).*conj(t_t(i));
            T_t_ref(i)=(sqrt(k1*k2)*t_t(i)+1).*conj(sqrt(k1*k2)*t_t(i)+1);

            gw2(i)=gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1*phi1v*a1'*gnd;

            gw1_diag=diag(phi1v*a1'*gnd*gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1);
            gw1_diag=diag(phi1v*a1'*gnd*gnd'*a1*phi1*D1*phi1v*a1'*a1*phi1*D1);

            fw1_diag=diag(phi1v*a2*phi2*D2*phi2v*a2'*phi1*D1*phi1v*a1'*gnd*gnd'*a1*phi1);
            fw=gw1_diag.'*(ones(1,size(phi1,1))') + (fw1_diag.'-gw1_diag.')*exp(-1i*(lambda1-w)*(time));

            fw2_t_ref(i)=fw;      
        end
    end


    %% Plotting


    T=k1*k2*t.*conj(t);
%     T2=k1*k2*t2.*conj(t2);
    r=t-1;

    tk=sqrt(k1*k2)*t;

    T_2port=(tk+1).*conj(tk+1);

    if showfig4==1
        tk_t=sqrt(k1*k2)*t_t;
        T_2port_t=(tk_t+1).*conj(tk_t+1);
    end

    if showfig2==1
        (fw2);
        g2_w=k1^2*k2^2./(T.^2).*fw2.*conj(fw2);
        g2_w_ref=abs(-k1*k2*fw2+4*(tk)+2).^2./(T_2port.^2)/4;
    end

    %g2_un=k1^2*k2^2.*g2w.*conj(g2w);%./(T.^2);

    %g2=k1^2*k2^2.*g2w.*conj(g2w)./(T.^2);

    if g3fig==1
        g3=k1^3*k2^3.*g3w.*conj(g3w)./(T.^3);
    end



    if showfig1==1
        figure(1)
        hold off
        %plot(W-wc,T2);
        ylim([0 1.2]);
        figure(1)
%         subplot(length(n),1,y)
        subplot(length(wc_list),1,z)
        plot(W-wc_list(y),T)
        hold on
        plot(W-wc_list(y),T_2port,'red')
        xlabel('Frequency')
        ylabel('Transmission Amplitude')
        lgd=legend('Transmission','Reflection');
        lgd.FontSize = 18;
    end

    if showfig2==1
        figure(2)
        %plot(W-wc,g2_w)
        subplot(length(wc_list),1,z)
        plot(W-wc_list(y),T,'blue')
        hold on
        plot(W-wc_list(y),T_2port,'red')
        plot(W-wc_list(y),g2_w,'green')
        plot(W-wc_list(y),g2_w_ref,'black')
        ylim([0 2])
        %plot(W-wc,g3)
        %plot(W-wc,T)
        legend('Transmission','Reflection')
        xlabel('Frequency')
        if g3fig==1
            g3=k1^3*k2^3.*g3w.*conj(g3w)./(T.^3);
            plot(W-wc_list(y),g3)
            lgd=legend('Transmitted g2','Transmitted g3');
            lgd.FontSize = 18;

        end

    end

    if showfig3==1
        figure(3)
        semilogy(W-wc_list(y),real(g2_w))
        ylim([0.1 1e5]);
        hold on
        semilogy(W-wc_list(y),g2_w_ref)
        hold on
        if g3fig==1
            semilogy(W-wc_list(y),g3)
        end
        lgd=legend('Transmission','Reflection')
        lgd.FontSize = 18;
    end

    %g2_w_ref=abs(-k1*k2*fw2-4*(tk)+2).^2./(T_2port.^2)/4;

    if showfig4==1

        g2_t_trans=k1^2*k2^2./(T_t_trans.^2).*abs(fw2_t).^2;
        g2_t_ref=1./(T_t_ref.^2).*abs(-k1*k2*fw2_t_ref+4*tk_t+2).^2/4;
        % why 5k (5000)?

        g2_t_trans_2=[flip(g2_t_trans),g2_t_trans];
        g2_t_ref_2=[flip(g2_t_ref),g2_t_ref];
        Time_2=[-flip(Time),Time];
        figure(4)
        plot(Time_2,g2_t_trans_2)
        title('g2(t)')
        hold on
        plot(Time_2,g2_t_ref_2)
        lgd=legend('Transmission','Reflection');
        lgd.FontSize = 18;
        xlabel('Time')
    end


end

% The program section to time. 
toc





