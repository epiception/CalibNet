echo 'nvcc'
/usr/local/cuda-8.0/bin/nvcc tf_auctionmatch_g.cu -o tf_auctionmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_30 
echo 'g++'

TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I ~/.conda/envs/my_tf1.3/lib/python2.7/site-packages/tensorflow/include -I ~/.conda/envs/my_tf1.3/lib/python2.7/site-packages/tensorflow/include/external/nsync/public  -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -L$TF_LIB -O2
