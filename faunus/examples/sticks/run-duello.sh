DUELLO="$HOME/.cargo/bin/duello"
RUST_LOG="Debug" $DUELLO scan \
	  -1 stick.xyz \
	  -2 stick.xyz \
	  --rmin 3.0 --rmax 40 --dr 0.5 \
	  --top duello-topology.yaml \
	  --resolution 0.3 \
	  --cutoff 1000 \
	  --molarity 0.01 \
	  --temperature 300
