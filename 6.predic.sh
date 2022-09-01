# echo '******our_test.py start******'
# python /data/qingyang/event_extration/DuEE_merge/data/DuEE1.0/our_test.py
# echo '******our_test.py end******'
# echo '******duee_1_data_prepare.py start******'
# python duee_1_data_prepare.py
# echo '******duee_1_data_prepare.py start******'
echo '******./3.sh start******'
./3-predict_trigger_for_sentence_EE.sh
echo '******./3.sh end******'
echo '******./4.sh start******'
./4-predict_role_for_sentence_EE.sh
echo '******./4.sh end******'
echo '******./5.sh start******'
./5-final.sh
echo '******./5.sh end******'