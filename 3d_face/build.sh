cd model/FaceBoxes
sh ./build_cpu_nms.sh
cd ../..

cd model/Sim3DR
sh ./build_sim3dr.sh
cd ../..

cd model/utils/asset
gcc -shared -Wall -O3 render.c -o render.so -fPIC
cd ../../..