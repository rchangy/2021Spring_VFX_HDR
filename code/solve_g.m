function solve_g(file_path)
    M = readmatrix(strcat(file_path, '/pixels.csv'));
    B = readmatrix(strcat(file_path,'/shutter.csv'));
    M_size = size(M);
    P = M_size(1)/3;
    
    Z_1 = M(1:P, :);
    Z_2 = M(P+1:2*P, :);
    Z_3 = M(2*P+1:3*P, :);
    lambda = 30;
    Z_min = 0;
    Z_max = 255;
    Z_mid = 0.5*(Z_min+Z_max);
    w = zeros(256, 1);
    for i=1:256
        if (i-1) <= Z_mid
            w(i) = i-1 - Z_min;
        else
            w(i) = Z_max - i+1;
        end
    end
    [g_1, lE_1] = gsolve(Z_1, B, lambda, w);
    [g_2, lE_2] = gsolve(Z_2, B, lambda, w);
    [g_3, lE_3] = gsolve(Z_3, B, lambda, w);
    g = [g_1'; g_2'; g_3';w'];
    
    writematrix(g, strcat(file_path, '/g.csv'))
    plot(g_1,[0:255], 'b')
    xlabel('log exposure')
    ylabel('pixel value')
    hold on
    plot(g_2,[0:255], 'g')
    plot(g_3,[0:255], 'r')
    hold off
end

function [g, lE]=gsolve(Z,B,l,w)
n = 256;
A = zeros(size(Z, 1)*size(Z, 2)+n+1, n+size(Z, 1));
b = zeros(size(A, 1), 1);
k = 1;
for i=1:size(Z,1)
    for j=1:size(Z, 2)
        wij=w(Z(i, j)+1);
        A(k, Z(i, j)+1) = wij;
        A(k, n+i) = -wij;
        b(k, 1) = wij * B(j);
        k = k+1;
    end
end
A(k, 129) = 1;
k=k+1;

for i=1:n-2
    A(k, i)=l*w(i+1);
    A(k, i+1)=-2*l*w(i+1);
    A(k, i+2) = l*w(i+1);
    k = k+1;
end

x=A\b;
g = x(1:n);
lE = x(n+1:size(x,1));
end