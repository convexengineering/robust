function [a,diff] = linearizeOpt(perts)
n = size(perts,2);
[input, counter]  = linearizeRecursive(n, 0, [], zeros(n,2^n));
output = zeros(1,counter);
diff = zeros(1,counter);
for i = 1:counter
    inputVec = input(:,i);
    output(i) = exp(inputVec'*perts');
end
M = [ones(1,counter);input]';
b = output';
f = @(x) sum(M*x - b);
a = fmincon(f, ones(n+1,1),-M,-b);

for i = 1:counter
    inputVec = input(:,i);
    diff(i) = a'*[1;inputVec] - output(i);
end