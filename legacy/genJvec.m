function [jvec] = genJvec(N)

if N==1
    jvec=[];
    return
end
ab=nchoosek(1:N,2)

a1 = ab(:,1);
a2 = ab(:,2);
% Convert both the numbers to strings
b1 = num2str(a1);
b2 = num2str(a2);
% Concatenate the two strings element wise
c1 = strcat(b1, b2);
% Convert the result back to a numeric matrix
jvec = str2num(c1)';


end