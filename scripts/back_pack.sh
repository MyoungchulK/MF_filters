st=$1
con=$2

source ../setup.sh
python3 back_est.py ${st} ${con} 0
python3 back_est_gof.py ${st} ${con} 0
python3 back_est_pseudo.py ${st} ${con} 0
python3 back_est.py ${st} ${con} 1
python3 back_est_gof.py ${st} ${con} 1
python3 back_est_pseudo.py ${st} ${con} 1
python3 upper_limit_summary.py ${st} ${con}



