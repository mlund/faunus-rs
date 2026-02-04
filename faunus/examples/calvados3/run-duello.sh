DUELLO="$HOME/.cargo/bin/duello"
RUST_LOG="Debug" $DUELLO scan \
	  -1 4lzt.xyz \
	  -2 4lzt.xyz \
	  --rmin 24.0 --rmax 70 --dr 0.5 \
	  --top duello-topology.yaml \
	  --resolution 0.5 \
	  --cutoff 1000 \
	  --molarity 0.05 \
	  --temperature 300
