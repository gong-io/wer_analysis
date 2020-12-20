#!/bin/bash

echo $#
if [ $# != 3 ]
then
    echo Usage: ./der_val.sh input_folder1 input_folder2 output_folder
else
    # Input and output folders
    v1=$1
    v2=$2
    results=$3

    # Sum of all files
    derSum=0
    missSum=0
    faSum=0
    spSum=0

    # Create an output for each file in the folders and a csv file containing each file's output as a line
    echo "filename,DER,MISS,FA,SPKR" > $output/der_comparison.csv
    for file in v1/*.rttm; do
        echo $file >> log
        filename=${file%.*}
        filename=${filename/v1\//}

        ./md-eval.pl -1 -c 0.25 -r $v1/$filename.rttm -s $v2/$filename.rttm 2 >  $results/${filename}_threshold.log > $results/${filename}_threshold.txt

        der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' $results/${filename}_threshold.txt)
        miss=$(grep -oP 'MISSED SPEAKER TIME.+\([ ]*\K[0-9]+([.][0-9]+)?'  $results/${filename}_threshold.txt)
        fa=$(grep -oP 'FALARM SPEAKER TIME.+\([ ]*\K[0-9]+([.][0-9]+)?'  $results/${filename}_threshold.txt)
        sp=$(grep -oP 'SPEAKER ERROR TIME.+\([ ]*\K[0-9]+([.][0-9]+)?'  $results/${filename}_threshold.txt)

        echo "${filename},$der,$miss,$fa,$sp" >> $output/der_comparison.csv

        # Add to the sum of all the files

        derSum=$(bc <<<  "scale=2; $derSum + $der")
        missSum=$(bc <<< "scale=2; $missSum + $miss")
        faSum=$(bc <<< "scale=2; $faSum + $fa")
        spSum=$(bc <<< "scale=2; $spSum + $sp")

    done

    echo "total,$derSum,$missSum,$faSum,$spSum" >> $output/der_comparison.csv
fi

