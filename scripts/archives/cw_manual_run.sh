station=$1
trig=$2
run=$3
run_w=500
ant_c=1
smear_l=20

source /home/mkim/analysis/MF_filters/setup.sh
echo "python3 cw_hist_livetime_test_smear_combine_cut.py ${station} ${trig} ${run} ${run_w} ${ant_c} ${smear_l}"
python3 cw_hist_livetime_test_smear_combine_cut.py ${station} ${trig} ${run} ${run_w} ${ant_c} ${smear_l}
