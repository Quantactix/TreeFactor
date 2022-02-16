cd data
echo '\n simulate data \n' 
Rscript simulate_data.r 
cd ../test/demo1
echo "\n run demo1 \n "
sh demo1.sh
cd ../demo2
echo "\n run demo2 \n "
sh demo2.sh
cd ../demo3
echo "\n run demo3 \n "
sh demo3.sh
