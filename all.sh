echo "*********** data_prepare *************"
# ./0-data_prepare_newTest.sh

echo "********** train start *************"
./1-train.sh

echo "********** predict start ***********"
./2-predict_postprocess.sh

echo "********** Done ***************"
