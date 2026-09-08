#!/bin/bash

source config.sh

mkdir -p collected_xvg_files

echo "Combining chunk data into .xvg files..."

RUN_FOLDER=("${MOLS[@]}")

for dir in "${RUN_FOLDER[@]}"; do
    if [ -d "$dir" ]; then
        echo "========================================="
        echo "Processing System: $dir"
        
        #Check for files
        for n in {2..6}; do
            if ls "$dir"/scratch/n${n}_chunk*_data.tmp 1> /dev/null 2>&1; then
                OUT_FILE_CCS="collected_xvg_files/${dir}_n${n}_ccs.xvg"
                OUT_FILE_BETA="collected_xvg_files/${dir}_n${n}_beta.xvg"
                
                # Sorting and processing the temp files into CCS and Beta files
                cat "$dir"/scratch/n${n}_chunk*_data.tmp \
                    | sort -k1,1n -k2,2n \
                    | awk -v dt="$DT" -v offset="$OFFSET" -v fccs="$OUT_FILE_CCS" -v fbeta="$OUT_FILE_BETA" '{
                        time_val = sprintf("%.4f", $1 * dt + offset);
                        print time_val, $3 > fccs;
                        if (NF >= 4) print time_val, $4 > fbeta;
                    }'
                
                echo " Created $OUT_FILE_CCS ($(wc -l < "$OUT_FILE_CCS") lines)"
                if [ -f "$OUT_FILE_BETA" ]; then
                    echo "Created $OUT_FILE_BETA ($(wc -l < "$OUT_FILE_BETA") lines)"
                fi
            else
                echo "ERR: No data found for N-mer $n in $dir"
            fi
        done
        
        # Also copy the main final_ccs.xvg if it exists
        if [ -f "${dir}/final_ccs.xvg" ]; then
            cp "${dir}/final_ccs.xvg" "collected_xvg_files/${dir}_final_ccs.xvg"
            echo "  Copied main final_ccs.xvg"
        fi
    fi
done

echo "Done!"
