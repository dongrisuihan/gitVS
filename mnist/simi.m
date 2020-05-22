load simi.mat;
num1=10;
num2=3;
similar=simi;
similar=similar+similar';
for i=1:num1
    similar(i,i)=0;
end
% figure(1);
% imshow(similar);
similar=similar/(max(similar(:)));
dis=-similar;
dis=dis-min(dis(:));
for i=1:num1
    dis(i,i)=0;
end
z=linkage(dis,'single');
dendrogram(z);
cst = cluster(z,'maxclust',num2);
f2c=zeros(num2,num1);
for i=1:num1
    f2c(cst(i),i)=1;
end

save('fine_to_coarse.mat','f2c');