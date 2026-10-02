#!/bin/bash

python3 assembly_stats.py \
    --name "Contigs" "HiC Scaffolds" \
	--title "Persea americana (Thille) Assembly" \
    --prefix "persea_americana_thille" \
    --genome-size 900m -n 12 -o /epool/edwin/pamericana/avoflow/hifiasm/thille/plots \
	-i /epool/edwin/pamericana/avoflow/hifiasm/thille/hifi.hic.gcluster/thille.hifiv3.asm.hic.p_ctg.fasta \
	   /epool/edwin/pamericana/avoflow/hifiasm/thille/hifi.hic.gcluster/yahs_thille.hifiv3.asm.hic.p_ctg/yahs.out_scaffolds_final.fa
