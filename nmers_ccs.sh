#!/bin/bash
#SBATCH --job-name=xtc_chunk
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00                
#SBATCH --array=0-199                   
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=bradyholdren@ucsb.edu
#SBATCH --mail-type=all

#Global Config
# ==============================================================================
CONDA_ENV="MDanalysis" #Needs mdanalysis, networkx, numpy and tqdm
source config.sh

module load miniconda/3.13
module load gromacs/2025.1

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
# ==============================================================================

ID=${SLURM_ARRAY_TASK_ID}

# Map ID to index
N=${N_MIN}
CHUNK=${ID}
if (( N_CHUNKS > 1 )); then
  N=$(( N_MIN + ID / N_CHUNKS ))
  CHUNK=$(( ID % N_CHUNKS ))
fi

# Calculate frame bounds
START_FRAME=$(( CHUNK * FRAME_CHUNK_SIZE ))
STOP_FRAME=$(( START_FRAME + FRAME_CHUNK_SIZE ))

# Read systems from config.sh
RUN_FOLDER=("${MOLS[@]}")

for CURRENT_RUN in "${RUN_FOLDER[@]}"; do
    echo "System: $CURRENT_RUN | Array Job: $ID | Chunk: $CHUNK"
    
    XTC_FILE="${CURRENT_RUN}/${XTC_NAME}"
    TPR_FILE="${CURRENT_RUN}/${TPR_NAME}"
    OUT_DIR_SCRATCH="${CURRENT_RUN}/scratch"
    DATA_OUT="${OUT_DIR_SCRATCH}/n${N}_chunk${CHUNK}_data.tmp"
    
    if [ ! -f "$XTC_FILE" ] || [ ! -f "$TPR_FILE" ]; then
        echo "WARNING: Missing .xtc or .tpr in $CURRENT_RUN. Skipping system."
        continue
    fi

    #Make scratch directory and clear previous run data
    mkdir -p "$OUT_DIR_SCRATCH"
    rm -f "$DATA_OUT"

    #Running find_nmers.py
    
    python "$PYTHON_SCRIPT" \
      --topol "$TPR_FILE" \
      --traj "$XTC_FILE" \
      --target "$N" \
      --start "$START_FRAME" \
      --stop "$STOP_FRAME" \
      --out "$OUT_DIR_SCRATCH/chunk${CHUNK}"

    #Calculating CCS
    for f in "$OUT_DIR_SCRATCH"/n${N}/chunk${CHUNK}_n${N}_frame*.pdb; do
        [ -e "$f" ] || { echo "No PDB frames found for chunk $CHUNK in $CURRENT_RUN."; continue; }
        
        RAW_SUFFIX=$(basename "$f" .pdb | awk -F '_frame' '{print $2}')
        TIME_PS=$(echo "$RAW_SUFFIX" | awk -F '_i' '{print $1}')
        INSTANCE_ID=$(echo "$RAW_SUFFIX" | awk -F '_i' '{print $2}')
        
        #Run IMPACT and parse CCS value
        CCS=$("$IMPACT_PATH" "$f" | grep "CCS 1" | awk '{print $(NF-1)}' || true)
        
        #Calculating Beta Strand Percentage using DSSP
        SS_XVG="${f%.pdb}_ss.xvg"
        gmx dssp -s "$f" -f "$f" -num "$SS_XVG" -o /dev/null -hmode dssp -clear > /dev/null 2>&1
        BETA_PCT=$(grep -v '^[#@]' "$SS_XVG" | awk '{tot=0; for(i=2;i<=11;i++) if(i!=3) tot+=$i; if(tot>0) printf "%.3f", $9/tot*100; else print "0.000"}' || true)
        rm -f "$SS_XVG"
        
        if [ -n "$CCS" ] && [ -n "$BETA_PCT" ]; then
            echo -e "${TIME_PS}\t${INSTANCE_ID}\t${CCS}\t${BETA_PCT}" >> "$DATA_OUT"
        fi
        
        rm -f "$f"
    done
done

echo "Array Job $ID finished successfully!"
