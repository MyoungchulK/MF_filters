st=$1
con=$2
pol=$3

python3 back_est.py ${st} ${con} ${pol}
python3 back_est_gof.py ${st} ${con} ${pol}
python3 back_est_pseudo.py ${st} ${con} ${pol}



