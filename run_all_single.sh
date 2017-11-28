#!/bin/bash -l
for MIN_CASES in 30
do
    for DATASET_NAME in BPI2012A.csv BPI2012O.csv BPI2012W.csv BPI2012W_no_dup.csv CreditRequirement.csv helpdesk.csv hospital_billing_977.csv traffic_fines_139.csv
    #for DATASET_NAME in bpic2017 traffic_fines
    do
        for METHOD in FA direct
        do
            for CLS_ENCODING in laststate combined
            do
                if [ $DATASET_NAME == "hospital_billing_977.csv" ] ; then
                    memory=20gb
                    niter=15
                elif [ $DATASET_NAME == "traffic_fines_139.csv" ] ; then
                    memory=20gb
                    niter=15
                else
                    memory=4gb
                    niter=40
                fi
                qsub -l mem=$memory -l walltime=40:00:00 -N job_"$DATASET_NAME"_"$METHOD"_"$CLS_ENCODING"_"$MIN_CASES" -v dataset=$DATASET_NAME,encoding=$CLS_ENCODING,min_cases=$MIN_CASES,niter=$niter run_$METHOD.sh
            done
        done
    done
done
